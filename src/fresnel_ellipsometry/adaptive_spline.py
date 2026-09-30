"""Adaptive KK-consistent B-spline refinement for a single sample."""

import numpy as np
import pandas as pd

from .dielectric import _build_spline_basis, build_fit_parameters
from .fitting import (
    _fit_measurement_chisqr, _fit_measurement_redchi, fit_bic, fit_spline,
    residual_search_metrics, residual_snapshot,
)
from .multistart import coarse_multistart_spline

def find_candidate_knots(params,spline,stack_model,measurement,bandwidth_nm,min_spacing_factor=1.5,residual_smooth_factor=1.0,n_regional=1,n_raw=1,n_individual=1):
    """Generate several candidate knot positions so one poor candidate cannot terminate refinement."""
    wl_nm = np.asarray(measurement.wl_nm,dtype=float)
    bandwidth_nm = np.asarray(bandwidth_nm,dtype=float)
    wl_step_nm = np.median(np.diff(np.sort(wl_nm))) if len(wl_nm) > 1 else 1.0
    metrics = residual_search_metrics(params,spline,stack_model,measurement,bandwidth_nm,residual_smooth_factor=residual_smooth_factor)
    existing_wl = spline.knot_wavelengths_nm
    candidates = []

    def allowed(idx):
        candidate_wl = float(wl_nm[idx])
        min_spacing_nm = max(min_spacing_factor*float(bandwidth_nm[idx]),wl_step_nm)
        return len(existing_wl) == 0 or np.min(np.abs(existing_wl-candidate_wl)) >= min_spacing_nm

    def add(metric,source,count):
        if count <= 0: return
        added = 0
        for idx in np.argsort(metric)[::-1]:
            if not allowed(idx): continue
            candidate_wl = float(wl_nm[idx])
            dedup_nm = max(float(bandwidth_nm[idx]),wl_step_nm)
            nearby = next((candidate for candidate in candidates if abs(candidate["wavelength_nm"]-candidate_wl) < dedup_nm),None)
            if nearby is not None:
                if source not in nearby["sources"]: nearby["sources"].append(source)
                continue
            candidates.append({"wavelength_nm":candidate_wl,"energy_eV":float(spline.hc_ev_nm/candidate_wl),"sources":[source],"regional_rms":float(metrics["regional_rms"][idx]),"raw_rms":float(metrics["raw_rms"][idx]),"individual_max":float(metrics["individual_max"][idx])})
            added += 1
            if added >= count: break

    add(metrics["regional_rms"],"regional",n_regional)
    add(metrics["raw_rms"],"raw",n_raw)
    add(metrics["individual_max"],"individual",n_individual)
    for candidate in candidates: candidate["source"] = "+".join(candidate.pop("sources"))
    candidates = sorted(candidates,key=lambda candidate:max(candidate["regional_rms"],candidate["raw_rms"],candidate["individual_max"]),reverse=True)
    metrics["candidates"] = candidates
    return metrics

def insert_knots(spline,knot_energy_eV):
    """Return a refined spline basis with one or several additional internal knots."""
    new_knots = np.atleast_1d(np.asarray(knot_energy_eV,dtype=float))
    all_knots = np.r_[spline.internal_knots,new_knots]
    all_knots = np.unique(np.round(all_knots,14))
    return _build_spline_basis(spline.wl_nm,all_knots,spline.degree,spline.kk_e_min,spline.kk_e_max,len(spline.E_kk),spline.hc_ev_nm)

def _basis_peak_wavelengths(spline):
    """Approximate the wavelength controlled most strongly by each retained B-spline coefficient."""
    return np.asarray(spline.wl_nm)[np.argmax(spline.B_meas,axis=0)]

def _seed_local_absorption(params,spline,stack_model,wavelengths_nm,seed_value,direction="both"):
    """Seed only spline coefficients whose basis peaks lie nearest the proposed sharp feature."""
    p = params.copy()
    unknown_name = stack_model.unknown_name
    peak_wl = _basis_peak_wavelengths(spline)
    wavelengths_nm = np.atleast_1d(np.asarray(wavelengths_nm,dtype=float))
    indices = sorted(set(int(np.argmin(np.abs(peak_wl-wl))) for wl in wavelengths_nm))

    if stack_model.is_isotropic:
        for j in indices:
            name = f"{unknown_name}_eps2_{j}"
            p[name].set(value=max(float(p[name].value),float(seed_value)))
        return p

    if direction in ("both","parallel"):
        for j in indices:
            name = f"{unknown_name}_eps2_o_{j}"
            p[name].set(value=max(float(p[name].value),float(seed_value)))
    if direction in ("both","perp"):
        for j in indices:
            name = f"{unknown_name}_eps2_e_{j}"
            p[name].set(value=max(float(p[name].value),float(seed_value)))
    return p

def _local_absorption_pinned(params,spline,stack_model,wavelengths_nm,tol=1e-7):
    """Detect whether every coefficient controlling a proposal remains effectively at its lower bound."""
    peak_wl = _basis_peak_wavelengths(spline)
    indices = sorted(set(int(np.argmin(np.abs(peak_wl-wl))) for wl in np.atleast_1d(wavelengths_nm)))
    unknown_name = stack_model.unknown_name
    if stack_model.is_isotropic:
        values = [params[f"{unknown_name}_eps2_{j}"].value for j in indices]
    else:
        values = [params[f"{unknown_name}_eps2_{direction}_{j}"].value for direction in ("o","e") for j in indices]
    return len(values) > 0 and float(np.max(values)) <= float(tol)

def _fit_proposal_multistart(base_params,spline,stack_model,measurement,seed_wavelengths_nm,seed_values=(0.05,0.25,1.0),max_nfev=600,reference_bic=None,required_bic_gain=0.0,include_directional_seeds=False):
    """Fit inherited coefficients first and escalate seeds only when that start is inadequate.

    The return value includes the number of nonlinear fits actually performed, making the
    speed/robustness tradeoff visible in ``candidate_trial_history``.
    """
    results = []

    def run(label,start):
        result = fit_spline(start,spline,stack_model,measurement,max_nfev=max_nfev)
        if np.isfinite(_fit_measurement_chisqr(result)): results.append((label,result))
        return result

    inherited = run("inherited",base_params)
    inherited_gain = np.inf if reference_bic is None else float(reference_bic-fit_bic(inherited))
    needs_seed = (not np.isfinite(_fit_measurement_chisqr(inherited)) or inherited_gain < required_bic_gain or
                  _local_absorption_pinned(inherited.params,spline,stack_model,seed_wavelengths_nm))
    if not needs_seed:
        return "inherited",inherited,len(results)

    # First seed all dielectric directions together. This is normally sufficient even
    # for uniaxial media and avoids tripling every bootstrap proposal pre-emptively.
    for seed in seed_values:
        start = _seed_local_absorption(base_params,spline,stack_model,seed_wavelengths_nm,seed,direction="both")
        seeded = run(f"seed={seed:g},both",start)
        if reference_bic is not None and np.isfinite(_fit_measurement_chisqr(seeded)) and reference_bic-fit_bic(seeded) >= required_bic_gain:
            break
    best = min(results,key=lambda item:_fit_measurement_chisqr(item[1])) if results else None
    best_gain = -np.inf if best is None or reference_bic is None else float(reference_bic-fit_bic(best[1]))

    # Separate parallel/perpendicular starts are the final fallback, not the default.
    if include_directional_seeds and stack_model.is_uniaxial and best_gain < required_bic_gain:
        directional_succeeded = False
        for seed in seed_values:
            for direction in ("parallel","perp"):
                start = _seed_local_absorption(base_params,spline,stack_model,seed_wavelengths_nm,seed,direction=direction)
                seeded = run(f"seed={seed:g},{direction}",start)
                if reference_bic is not None and np.isfinite(_fit_measurement_chisqr(seeded)) and reference_bic-fit_bic(seeded) >= required_bic_gain:
                    directional_succeeded = True
                    break
            if directional_succeeded:
                break

    if len(results) == 0: raise RuntimeError("Every proposal start returned a non-finite chi-square.")
    label,result = min(results,key=lambda item:_fit_measurement_chisqr(item[1]))
    return label,result,len(results)

def _bootstrap_knot_sets(center_wl_nm,spline,measurement,bandwidth_nm,min_spacing_factor=1.0,bundle_sizes=(2,3)):
    """Create local 2- and 3-knot bundles around a difficult residual feature."""
    wl_nm = np.asarray(measurement.wl_nm,dtype=float)
    bandwidth_nm = np.asarray(bandwidth_nm,dtype=float)
    wl_step_nm = np.median(np.diff(np.sort(wl_nm))) if len(wl_nm) > 1 else 1.0
    idx = int(np.argmin(np.abs(wl_nm-float(center_wl_nm))))
    spacing_nm = max(min_spacing_factor*float(bandwidth_nm[idx]),1.5*wl_step_nm)
    wl_min,wl_max = float(np.min(wl_nm)),float(np.max(wl_nm))
    existing = np.asarray(spline.knot_wavelengths_nm,dtype=float)
    bundles = []

    for size in bundle_sizes:
        if int(size) == 2:
            offsets = np.array([-0.5,0.5])*spacing_nm
        elif int(size) == 3:
            offsets = np.array([-1.0,0.0,1.0])*spacing_nm
        else:
            offsets = (np.arange(int(size))-(int(size)-1)/2.0)*spacing_nm
        proposed = np.clip(float(center_wl_nm)+offsets,wl_min,wl_max)
        proposed = np.unique(proposed)
        keep = []
        for wl in proposed:
            min_spacing_nm = max(0.75*float(np.interp(wl,wl_nm,bandwidth_nm)),wl_step_nm)
            if len(existing) == 0 or np.min(np.abs(existing-wl)) >= min_spacing_nm: keep.append(float(wl))
        if len(keep) >= 2: bundles.append(np.asarray(keep,dtype=float))
    return bundles

def adaptive_fit_spline(params,spline,stack_model,measurement,bandwidth_nm,max_iterations=12,max_free_nodes=24,max_nfev=100000,residual_threshold=1.5,individual_residual_threshold=4.0,bic_improvement=2.0,min_spacing_factor=1.5,residual_smooth_factor=1.0,trial_max_nfev=600,bootstrap=True,max_single_candidates=3,initial_multistart=False,multistart_n_starts=10,multistart_screen_max_nfev=250,multistart_n_finalists=3,verbose=False,show_progress=True):
    """Hierarchical adaptive fit with optional robust coarse multistart initialization."""
    if initial_multistart:
        fit_result,multistart_history = coarse_multistart_spline(
            params,spline,stack_model,measurement,
            n_starts=multistart_n_starts,screen_max_nfev=multistart_screen_max_nfev,
            n_finalists=multistart_n_finalists,max_nfev=max_nfev,
            show_progress=show_progress,verbose=verbose,
        )
        initial_mode = "initial-multistart"
    else:
        if show_progress: print(f"Spline iteration 0/{int(max_iterations)}",end="\r",flush=True)
        fit_result = fit_spline(params,spline,stack_model,measurement,max_nfev=max_nfev)
        multistart_history = None
        initial_mode = "initial"
    fit_result.initial_multistart_history = multistart_history
    history = [{"iteration":0,"mode":initial_mode,"free_nodes":spline.n_free_nodes,"chisqr":_fit_measurement_chisqr(fit_result),"redchi":_fit_measurement_redchi(fit_result),"bic":fit_bic(fit_result),"added_wavelength_nm":np.nan,"n_added_knots":0,"accepted":True}]
    snapshots = [residual_snapshot(fit_result,spline,stack_model,measurement,0)]

    for iteration in range(1,int(max_iterations)+1):
        if show_progress: print(f"Spline iteration {iteration}/{int(max_iterations)}",end="\r",flush=True)
        search = find_candidate_knots(fit_result.params,spline,stack_model,measurement,bandwidth_nm,min_spacing_factor=min_spacing_factor,residual_smooth_factor=residual_smooth_factor,n_regional=1,n_raw=1,n_individual=1)
        if verbose: print(f'Iteration {iteration}: regional={search["max_regional_rms"]:.2f}σ, individual={search["max_individual_residual"]:.2f}σ.')

        if spline.n_free_nodes >= max_free_nodes:
            if verbose: print(f"Stopping: reached {max_free_nodes} spline coefficients.")
            break
        if search["max_regional_rms"] <= residual_threshold and search["max_individual_residual"] <= individual_residual_threshold:
            if verbose: print("Stopping: residual criteria satisfied.")
            break
        if len(search["candidates"]) == 0:
            if verbose: print("Stopping: no allowed knot candidate.")
            break

        # ----- Stage 1: normal single-knot look-ahead -----
        single_trials = []
        current_bic = fit_bic(fit_result)
        for candidate in search["candidates"][:max(1,int(max_single_candidates))]:
            proposal_spline = insert_knots(spline,[candidate["energy_eV"]])
            proposal_params = build_fit_parameters(stack_model,proposal_spline,previous_result=fit_result,old_spline=spline)
            label,proposal_result,n_starts = _fit_proposal_multistart(proposal_params,proposal_spline,stack_model,measurement,[candidate["wavelength_nm"]],seed_values=(0.10,),max_nfev=trial_max_nfev,reference_bic=current_bic,required_bic_gain=bic_improvement,include_directional_seeds=False)
            bic_gain = current_bic-fit_bic(proposal_result)
            single_trials.append({"mode":"single","wavelengths_nm":[candidate["wavelength_nm"]],"candidate_source":candidate["source"],"seed":label,"n_starts":n_starts,"spline":proposal_spline,"result":proposal_result,"bic_gain":bic_gain})

        best_single = max(single_trials,key=lambda trial:trial["bic_gain"])
        chosen = best_single if best_single["bic_gain"] >= bic_improvement else None

        # ----- Stage 2: resonance bootstrap only when ordinary refinement cannot improve a still-bad fit -----
        bootstrap_trials = []
        if chosen is None and bootstrap:
            centers = [candidate["wavelength_nm"] for candidate in search["candidates"][:2]]
            # Two-knot bundles are tested first. Three-knot bundles are considered only
            # when every cheaper two-knot proposal fails the BIC gate.
            for bundle_size in (2,3):
                stage_trials = []
                for center in centers:
                    bundles = _bootstrap_knot_sets(center,spline,measurement,bandwidth_nm,min_spacing_factor=1.0,bundle_sizes=(bundle_size,))
                    for bundle_wl in bundles:
                        if spline.n_free_nodes+len(bundle_wl) > max_free_nodes: continue
                        proposal_spline = insert_knots(spline,spline.hc_ev_nm/bundle_wl)
                        proposal_params = build_fit_parameters(stack_model,proposal_spline,previous_result=fit_result,old_spline=spline)
                        label,proposal_result,n_starts = _fit_proposal_multistart(proposal_params,proposal_spline,stack_model,measurement,bundle_wl,seed_values=(0.03,0.20,1.0),max_nfev=trial_max_nfev,reference_bic=current_bic,required_bic_gain=bic_improvement,include_directional_seeds=True)
                        bic_gain = current_bic-fit_bic(proposal_result)
                        mode = f"bootstrap-{len(bundle_wl)}"
                        trial = {"mode":mode,"wavelengths_nm":bundle_wl.tolist(),"candidate_source":"sharp-feature bootstrap","seed":label,"n_starts":n_starts,"spline":proposal_spline,"result":proposal_result,"bic_gain":bic_gain}
                        stage_trials.append(trial)
                        bootstrap_trials.append(trial)
                passing = [trial for trial in stage_trials if trial["bic_gain"] >= bic_improvement]
                if passing:
                    chosen = max(passing,key=lambda trial:trial["bic_gain"])
                    break

        # If neither a single knot nor a bundle improves BIC enough, stop.
        if chosen is None:
            best_gain = max([trial["bic_gain"] for trial in single_trials]+[trial["bic_gain"] for trial in bootstrap_trials])
            if verbose: print(f"Stopping: best BIC improvement was {best_gain:.2f}.")
            break

        # Fully converge the selected proposal before accepting it.
        proposal_result = fit_spline(chosen["result"].params,chosen["spline"],stack_model,measurement,max_nfev=max_nfev)
        final_bic_gain = fit_bic(fit_result)-fit_bic(proposal_result)
        accepted = bool(proposal_result.success and final_bic_gain >= bic_improvement)
        added_wl = np.asarray(chosen["wavelengths_nm"],dtype=float)
        history.append({"iteration":iteration,"mode":chosen["mode"],"free_nodes":chosen["spline"].n_free_nodes,"chisqr":_fit_measurement_chisqr(proposal_result),"redchi":_fit_measurement_redchi(proposal_result),"bic":fit_bic(proposal_result),"added_wavelength_nm":float(np.mean(added_wl)),"n_added_knots":len(added_wl),"accepted":accepted})
        if verbose: print(f'Selected {chosen["mode"]} at {", ".join(f"{wl:.1f}" for wl in added_wl)} nm; ΔBIC={final_bic_gain:.2f}.')

        if not accepted: break
        spline,fit_result = chosen["spline"],proposal_result
        snapshots.append(residual_snapshot(fit_result,spline,stack_model,measurement,iteration))
    fit_result.initial_multistart_history = multistart_history
    if show_progress: print()
    return fit_result,spline,pd.DataFrame(history),snapshots

