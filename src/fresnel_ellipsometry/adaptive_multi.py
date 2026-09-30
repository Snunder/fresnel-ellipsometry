"""Adaptive spline and physical-model refinement for shared-dielectric multi-sample fits."""

import numpy as np
import pandas as pd

from .models import configure_multifit_thickness_parameters
from .dielectric import _canonical_physical_kind, add_physical_term
from .fitting import _fit_measurement_chisqr, _fit_measurement_redchi, fit_bic, residual_search_metrics
from .adaptive_spline import _local_absorption_pinned, _seed_local_absorption, insert_knots
from .adaptive_physical import _physical_seeds
from .multifit import (
    _validate_multifit_datasets, build_multi_fit_parameters, build_multi_physical_parameters,
    fit_spline_multi, fit_physical_multi,
)
from .multistart import coarse_multistart_spline_multi
from .fitting import ellipsometry_residuals

def _multi_metric_grid(params,optical_model,datasets,residual_smooth_factor=1.0):
    """Pool wavelength-resolved residual diagnostics from flakes with possibly different grids."""
    grid = np.asarray(optical_model.wl_nm,dtype=float)
    regional,raw,individual,bandwidth_curves = [],[],[],[]
    for dataset in _validate_multifit_datasets(datasets):
        m = residual_search_metrics(params,optical_model,dataset.stack_model,dataset.measurement,dataset.bandwidth_nm,residual_smooth_factor=residual_smooth_factor)
        wl = np.asarray(dataset.measurement.wl_nm,dtype=float)
        inside = (grid >= np.min(wl)) & (grid <= np.max(wl))
        def interp(values):
            out = np.full(grid.shape,np.nan,dtype=float)
            out[inside] = np.interp(grid[inside],wl,np.asarray(values,dtype=float))
            return out
        regional.append(interp(m["regional_rms"]))
        raw.append(interp(m["raw_rms"]))
        individual.append(interp(m["individual_max"]))
        bandwidth_curves.append(interp(dataset.bandwidth_nm))
    def rms_nan(arrays):
        a = np.stack(arrays)
        valid = np.sum(np.isfinite(a),axis=0)
        total = np.nansum(a*a,axis=0)
        return np.sqrt(total/np.maximum(valid,1))
    reg = rms_nan(regional)
    raw_r = rms_nan(raw)
    ind = np.nanmax(np.stack(individual),axis=0)
    bw = np.nanmedian(np.stack(bandwidth_curves),axis=0)
    finite_bw = bw[np.isfinite(bw)]
    fallback = float(np.median(finite_bw)) if finite_bw.size else 1.0
    bw = np.where(np.isfinite(bw),bw,fallback)
    return {"grid_nm":grid,"bandwidth_nm":bw,"regional_rms":reg,"raw_rms":raw_r,"individual_max":ind,
            "max_regional_rms":float(np.nanmax(reg)),"max_raw_rms":float(np.nanmax(raw_r)),"max_individual_residual":float(np.nanmax(ind))}

def find_candidate_knots_multi(params,spline,datasets,min_spacing_factor=1.5,residual_smooth_factor=1.0,n_regional=1,n_raw=1,n_individual=1):
    metrics = _multi_metric_grid(params,spline,datasets,residual_smooth_factor=residual_smooth_factor)
    wl_nm,bandwidth_nm = metrics["grid_nm"],metrics["bandwidth_nm"]
    wl_step_nm = np.median(np.diff(np.sort(wl_nm))) if len(wl_nm) > 1 else 1.0
    existing_wl = spline.knot_wavelengths_nm
    candidates = []
    def allowed(idx):
        candidate_wl = float(wl_nm[idx])
        min_spacing_nm = max(min_spacing_factor*float(bandwidth_nm[idx]),wl_step_nm)
        return len(existing_wl) == 0 or np.min(np.abs(existing_wl-candidate_wl)) >= min_spacing_nm
    def add(metric,source,count):
        if count <= 0: return
        added = 0
        for idx in np.argsort(np.nan_to_num(metric,nan=-np.inf))[::-1]:
            if not np.isfinite(metric[idx]) or not allowed(idx): continue
            candidate_wl = float(wl_nm[idx])
            dedup_nm = max(float(bandwidth_nm[idx]),wl_step_nm)
            nearby = next((candidate for candidate in candidates if abs(candidate["wavelength_nm"]-candidate_wl) < dedup_nm),None)
            if nearby is not None:
                if source not in nearby["sources"]: nearby["sources"].append(source)
                continue
            candidates.append({"wavelength_nm":candidate_wl,"energy_eV":float(spline.hc_ev_nm/candidate_wl),"sources":[source],
                               "regional_rms":float(metrics["regional_rms"][idx]),"raw_rms":float(metrics["raw_rms"][idx]),"individual_max":float(metrics["individual_max"][idx])})
            added += 1
            if added >= count: break
    add(metrics["regional_rms"],"regional",n_regional)
    add(metrics["raw_rms"],"raw",n_raw)
    add(metrics["individual_max"],"individual",n_individual)
    for candidate in candidates: candidate["source"] = "+".join(candidate.pop("sources"))
    metrics["candidates"] = sorted(candidates,key=lambda candidate:max(candidate["regional_rms"],candidate["raw_rms"],candidate["individual_max"]),reverse=True)
    return metrics

def _fit_proposal_multistart_multi(base_params,spline,datasets,seed_wavelengths_nm,seed_values=(0.05,0.25,1.0),max_nfev=600,reference_bic=None,required_bic_gain=0.0,include_directional_seeds=False):
    datasets = _validate_multifit_datasets(datasets)
    reference_stack = datasets[0].stack_model
    results = []
    def run(label,start):
        result = fit_spline_multi(start,spline,datasets,max_nfev=max_nfev)
        if np.isfinite(_fit_measurement_chisqr(result)): results.append((label,result))
        return result
    inherited = run("inherited",base_params)
    inherited_gain = np.inf if reference_bic is None else float(reference_bic-fit_bic(inherited))
    needs_seed = (not np.isfinite(_fit_measurement_chisqr(inherited)) or inherited_gain < required_bic_gain or
                  _local_absorption_pinned(inherited.params,spline,reference_stack,seed_wavelengths_nm))
    if not needs_seed: return "inherited",inherited,len(results)
    for seed in seed_values:
        start = _seed_local_absorption(base_params,spline,reference_stack,seed_wavelengths_nm,seed,direction="both")
        seeded = run(f"seed={seed:g},both",start)
        if reference_bic is not None and np.isfinite(_fit_measurement_chisqr(seeded)) and reference_bic-fit_bic(seeded) >= required_bic_gain: break
    best = min(results,key=lambda item:_fit_measurement_chisqr(item[1])) if results else None
    best_gain = -np.inf if best is None or reference_bic is None else float(reference_bic-fit_bic(best[1]))
    if include_directional_seeds and reference_stack.is_uniaxial and best_gain < required_bic_gain:
        directional_succeeded = False
        for seed in seed_values:
            for direction in ("parallel","perp"):
                start = _seed_local_absorption(base_params,spline,reference_stack,seed_wavelengths_nm,seed,direction=direction)
                seeded = run(f"seed={seed:g},{direction}",start)
                if reference_bic is not None and np.isfinite(_fit_measurement_chisqr(seeded)) and reference_bic-fit_bic(seeded) >= required_bic_gain:
                    directional_succeeded = True
                    break
            if directional_succeeded: break
    if len(results) == 0: raise RuntimeError("Every multi-flake proposal start returned a non-finite chi-square.")
    label,result = min(results,key=lambda item:_fit_measurement_chisqr(item[1]))
    return label,result,len(results)

def _bootstrap_knot_sets_multi(center_wl_nm,spline,datasets,min_spacing_factor=1.0,bundle_sizes=(2,3)):
    grid = np.asarray(spline.wl_nm,dtype=float)
    bandwidth_curves = []
    for dataset in datasets:
        curve = np.full(grid.shape,np.nan,dtype=float)
        wl = np.asarray(dataset.measurement.wl_nm,dtype=float)
        inside = (grid >= wl.min()) & (grid <= wl.max())
        curve[inside] = np.interp(grid[inside],wl,np.asarray(dataset.bandwidth_nm,dtype=float))
        bandwidth_curves.append(curve)
    bw = np.nanmedian(np.stack(bandwidth_curves),axis=0)
    finite = bw[np.isfinite(bw)]
    bw = np.where(np.isfinite(bw),bw,float(np.median(finite)) if finite.size else 1.0)
    wl_step_nm = np.median(np.diff(np.sort(grid))) if len(grid) > 1 else 1.0
    idx = int(np.argmin(np.abs(grid-float(center_wl_nm))))
    spacing_nm = max(min_spacing_factor*float(bw[idx]),1.5*wl_step_nm)
    wl_min,wl_max = float(np.min(grid)),float(np.max(grid))
    existing = np.asarray(spline.knot_wavelengths_nm,dtype=float)
    bundles = []
    for size in bundle_sizes:
        offsets = np.array([-0.5,0.5])*spacing_nm if int(size) == 2 else (np.array([-1.0,0.0,1.0])*spacing_nm if int(size) == 3 else (np.arange(int(size))-(int(size)-1)/2.0)*spacing_nm)
        proposed = np.unique(np.clip(float(center_wl_nm)+offsets,wl_min,wl_max))
        keep = []
        for wl in proposed:
            min_spacing_nm = max(0.75*float(np.interp(wl,grid,bw)),wl_step_nm)
            if len(existing) == 0 or np.min(np.abs(existing-wl)) >= min_spacing_nm: keep.append(float(wl))
        if len(keep) >= 2: bundles.append(np.asarray(keep,dtype=float))
    return bundles

def residual_snapshot_multi(result,optical_model,datasets,iteration):
    stored = {}
    for dataset in datasets:
        residual = ellipsometry_residuals(result.params,optical_model,dataset.stack_model,dataset.measurement)
        stored[dataset.label] = {
            "psi_residual_deg":{aoi:values.copy() for aoi,values in residual["psi_deg"].items()},
            "delta_residual_deg":{aoi:values.copy() for aoi,values in residual["delta_deg"].items()},
        }
    return {"iteration":int(iteration),"datasets":stored}

def adaptive_fit_spline_multi(params,spline,datasets,max_iterations=12,max_free_nodes=24,max_nfev=100000,residual_threshold=1.5,individual_residual_threshold=4.0,bic_improvement=2.0,min_spacing_factor=1.5,residual_smooth_factor=1.0,trial_max_nfev=600,bootstrap=True,max_single_candidates=3,initial_multistart=False,multistart_n_starts=10,multistart_screen_max_nfev=250,multistart_n_finalists=3,verbose=False,show_progress=True):
    """Adaptive shared-dielectric B-spline fit with optional coarse multistart initialization."""
    datasets = configure_multifit_thickness_parameters(_validate_multifit_datasets(datasets))
    if initial_multistart:
        fit_result,multistart_history = coarse_multistart_spline_multi(
            params,spline,datasets,
            n_starts=multistart_n_starts,screen_max_nfev=multistart_screen_max_nfev,
            n_finalists=multistart_n_finalists,max_nfev=max_nfev,
            show_progress=show_progress,verbose=verbose,
        )
        initial_mode = "initial-multistart"
    else:
        if show_progress: print(f"Multi-flake spline iteration 0/{int(max_iterations)}",end="\r",flush=True)
        fit_result = fit_spline_multi(params,spline,datasets,max_nfev=max_nfev)
        multistart_history = None
        initial_mode = "initial"
    fit_result.initial_multistart_history = multistart_history
    history = [{"iteration":0,"mode":initial_mode,"free_nodes":spline.n_free_nodes,"chisqr":_fit_measurement_chisqr(fit_result),"redchi":_fit_measurement_redchi(fit_result),"bic":fit_bic(fit_result),"added_wavelength_nm":np.nan,"n_added_knots":0,"accepted":True}]
    snapshots = [residual_snapshot_multi(fit_result,spline,datasets,0)]
    reference_stack = datasets[0].stack_model
    for iteration in range(1,int(max_iterations)+1):
        if show_progress: print(f"Multi-flake spline iteration {iteration}/{int(max_iterations)}",end="\r",flush=True)
        search = find_candidate_knots_multi(fit_result.params,spline,datasets,min_spacing_factor=min_spacing_factor,residual_smooth_factor=residual_smooth_factor,n_regional=1,n_raw=1,n_individual=1)
        if verbose: print(f'Iteration {iteration}: regional={search["max_regional_rms"]:.2f}σ, individual={search["max_individual_residual"]:.2f}σ.')
        if spline.n_free_nodes >= max_free_nodes: break
        if search["max_regional_rms"] <= residual_threshold and search["max_individual_residual"] <= individual_residual_threshold: break
        if len(search["candidates"]) == 0: break
        current_bic = fit_bic(fit_result)
        single_trials = []
        for candidate in search["candidates"][:max(1,int(max_single_candidates))]:
            proposal_spline = insert_knots(spline,[candidate["energy_eV"]])
            proposal_params = build_multi_fit_parameters(datasets,proposal_spline,previous_result=fit_result,old_spline=spline)
            label,proposal_result,n_starts = _fit_proposal_multistart_multi(proposal_params,proposal_spline,datasets,[candidate["wavelength_nm"]],seed_values=(0.10,),max_nfev=trial_max_nfev,reference_bic=current_bic,required_bic_gain=bic_improvement,include_directional_seeds=False)
            bic_gain = current_bic-fit_bic(proposal_result)
            single_trials.append({"mode":"single","wavelengths_nm":[candidate["wavelength_nm"]],"candidate_source":candidate["source"],"seed":label,"n_starts":n_starts,"spline":proposal_spline,"result":proposal_result,"bic_gain":bic_gain})
        best_single = max(single_trials,key=lambda trial:trial["bic_gain"])
        chosen = best_single if best_single["bic_gain"] >= bic_improvement else None
        bootstrap_trials = []
        if chosen is None and bootstrap:
            centers = [candidate["wavelength_nm"] for candidate in search["candidates"][:2]]
            for bundle_size in (2,3):
                stage_trials = []
                for center in centers:
                    for bundle_wl in _bootstrap_knot_sets_multi(center,spline,datasets,min_spacing_factor=1.0,bundle_sizes=(bundle_size,)):
                        if spline.n_free_nodes+len(bundle_wl) > max_free_nodes: continue
                        proposal_spline = insert_knots(spline,spline.hc_ev_nm/bundle_wl)
                        proposal_params = build_multi_fit_parameters(datasets,proposal_spline,previous_result=fit_result,old_spline=spline)
                        label,proposal_result,n_starts = _fit_proposal_multistart_multi(proposal_params,proposal_spline,datasets,bundle_wl,seed_values=(0.03,0.20,1.0),max_nfev=trial_max_nfev,reference_bic=current_bic,required_bic_gain=bic_improvement,include_directional_seeds=True)
                        bic_gain = current_bic-fit_bic(proposal_result)
                        trial = {"mode":f"bootstrap-{len(bundle_wl)}","wavelengths_nm":bundle_wl.tolist(),"candidate_source":"sharp-feature bootstrap","seed":label,"n_starts":n_starts,"spline":proposal_spline,"result":proposal_result,"bic_gain":bic_gain}
                        stage_trials.append(trial); bootstrap_trials.append(trial)
                passing = [trial for trial in stage_trials if trial["bic_gain"] >= bic_improvement]
                if passing:
                    chosen = max(passing,key=lambda trial:trial["bic_gain"]); break
        if chosen is None: break
        proposal_result = fit_spline_multi(chosen["result"].params,chosen["spline"],datasets,max_nfev=max_nfev)
        final_bic_gain = fit_bic(fit_result)-fit_bic(proposal_result)
        accepted = bool(proposal_result.success and final_bic_gain >= bic_improvement)
        added_wl = np.asarray(chosen["wavelengths_nm"],dtype=float)
        history.append({"iteration":iteration,"mode":chosen["mode"],"free_nodes":chosen["spline"].n_free_nodes,"chisqr":_fit_measurement_chisqr(proposal_result),"redchi":_fit_measurement_redchi(proposal_result),"bic":fit_bic(proposal_result),"added_wavelength_nm":float(np.mean(added_wl)),"n_added_knots":len(added_wl),"accepted":accepted})
        if verbose: print(f'Selected {chosen["mode"]} at {", ".join(f"{wl:.1f}" for wl in added_wl)} nm; ΔBIC={final_bic_gain:.2f}.')
        if not accepted: break
        spline,fit_result = chosen["spline"],proposal_result
        snapshots.append(residual_snapshot_multi(fit_result,spline,datasets,iteration))
    fit_result.initial_multistart_history = multistart_history
    if show_progress: print()
    return fit_result,spline,pd.DataFrame(history),snapshots

def _rank_physical_centers_multi(params,model,datasets,n_centers=3,residual_smooth_factor=0.5):
    """Return separated pooled residual peaks used as nonlinear physical-model seeds."""
    metrics = _multi_metric_grid(params,model,datasets,residual_smooth_factor=residual_smooth_factor)
    wl = np.asarray(metrics["grid_nm"],dtype=float)
    score = np.maximum(np.asarray(metrics["regional_rms"]),np.asarray(metrics["individual_max"]))
    local_bandwidth = np.asarray(metrics["bandwidth_nm"],dtype=float)
    centers = []
    for idx in np.argsort(np.nan_to_num(score,nan=-np.inf))[::-1]:
        if not np.isfinite(score[idx]): continue
        candidate = float(wl[idx])
        separation = max(2.0*float(local_bandwidth[idx]),0.04*float(np.ptp(wl)))
        if all(abs(candidate-existing) >= separation for existing in centers):
            centers.append(candidate)
        if len(centers) >= int(n_centers): break
    if not centers: centers = [float(np.median(wl))]
    return centers,metrics

def adaptive_fit_physical_multi(params,model,datasets,candidate_types=("cauchy","sellmeier_uv","sellmeier_ir","drude","lorentz","harmonic","gaussian","tauc_lorentz","cody_lorentz"),max_terms=5,max_nfev=100000,trial_max_nfev=600,screen_max_nfev=180,n_family_finalists=4,bic_improvement=6.0,residual_threshold=1.5,individual_residual_threshold=4.0,residual_smooth_factor=0.5,n_centers=3,verbose=False,show_progress=True):
    """Sequential BIC selection of one physical dielectric tensor shared by all flakes."""
    datasets = configure_multifit_thickness_parameters(_validate_multifit_datasets(datasets))
    candidate_types = tuple(_canonical_physical_kind(kind) for kind in candidate_types)
    allowed = {"cauchy","sellmeier_uv","sellmeier_ir","drude","lorentz","harmonic","gaussian","tauc_lorentz","cody_lorentz"}
    invalid = sorted(set(candidate_types)-allowed)
    if invalid: raise ValueError(f"Unsupported physical candidate type(s): {invalid}.")
    if not candidate_types: raise ValueError("candidate_types may not be empty.")
    if show_progress: print(f"Multi-flake physical iteration 0/{int(max_terms)}",end="\r",flush=True)
    fit_result = fit_physical_multi(params,model,datasets,max_nfev=max_nfev)
    history = [{"iteration":0,"mode":"baseline","added_type":"none","model":model.description,"n_components":0,"chisqr":_fit_measurement_chisqr(fit_result),"redchi":_fit_measurement_redchi(fit_result),"bic":fit_bic(fit_result),"bic_gain":np.nan,"accepted":True}]
    snapshots = [residual_snapshot_multi(fit_result,model,datasets,0)]
    for iteration in range(1,int(max_terms)+1):
        if show_progress: print(f"Multi-flake physical iteration {iteration}/{int(max_terms)}",end="\r",flush=True)
        centers,search = _rank_physical_centers_multi(fit_result.params,model,datasets,n_centers=n_centers,residual_smooth_factor=residual_smooth_factor)
        if verbose: print(f'Physical iteration {iteration}: regional={search["max_regional_rms"]:.2f}σ, individual={search["max_individual_residual"]:.2f}σ.')
        if search["max_regional_rms"] <= residual_threshold and search["max_individual_residual"] <= individual_residual_threshold:
            if verbose: print("Stopping: residual criteria satisfied.")
            break
        current_bic = fit_bic(fit_result)
        family_records = []
        for kind in candidate_types:
            if kind == "cauchy" and model.background == "cauchy": continue
            proposal_model = add_physical_term(model,kind)
            seeds = _physical_seeds(kind,centers,model)
            seed_label,seed = seeds[0]
            proposal_params = build_multi_physical_parameters(datasets,proposal_model,previous_result=fit_result,new_term_seed=seed)
            trial = fit_physical_multi(proposal_params,proposal_model,datasets,max_nfev=min(int(screen_max_nfev),int(trial_max_nfev)))
            gain = current_bic-fit_bic(trial)
            family_records.append({"kind":kind,"model":proposal_model,"seeds":seeds,"best_seed":seed_label,"best_result":trial,"bic_gain":gain})
        if not family_records:
            if verbose: print("Stopping: no candidate families remain.")
            break
        finalists = sorted(family_records,key=lambda item:item["bic_gain"],reverse=True)[:max(1,int(n_family_finalists))]
        for record in finalists:
            for seed_label,seed in record["seeds"][1:]:
                proposal_params = build_multi_physical_parameters(datasets,record["model"],previous_result=fit_result,new_term_seed=seed)
                trial = fit_physical_multi(proposal_params,record["model"],datasets,max_nfev=trial_max_nfev)
                gain = current_bic-fit_bic(trial)
                if np.isfinite(_fit_measurement_chisqr(trial)) and gain > record["bic_gain"]:
                    record.update({"best_seed":seed_label,"best_result":trial,"bic_gain":gain})
        chosen = max(family_records,key=lambda proposal:proposal["bic_gain"])
        if not np.isfinite(_fit_measurement_chisqr(chosen["best_result"])) or chosen["bic_gain"] < bic_improvement:
            if verbose: print(f'Stopping: best candidate {chosen["kind"]} improved BIC by {chosen["bic_gain"]:.2f}.')
            break
        converged = fit_physical_multi(chosen["best_result"].params,chosen["model"],datasets,max_nfev=max_nfev)
        final_gain = current_bic-fit_bic(converged)
        accepted = bool(converged.success and np.isfinite(_fit_measurement_chisqr(converged)) and final_gain >= bic_improvement)
        component_count = chosen["model"].n_terms+int(chosen["model"].background != "constant")
        history.append({"iteration":iteration,"mode":"physical","added_type":chosen["kind"],"model":chosen["model"].description,"n_components":component_count,"chisqr":_fit_measurement_chisqr(converged),"redchi":_fit_measurement_redchi(converged),"bic":fit_bic(converged),"bic_gain":final_gain,"accepted":accepted})
        if verbose: print(f'Selected {chosen["kind"]}; ΔBIC={final_gain:.2f}.')
        if not accepted: break
        model,fit_result = chosen["model"],converged
        snapshots.append(residual_snapshot_multi(fit_result,model,datasets,iteration))
    if show_progress: print()
    return fit_result,model,pd.DataFrame(history),snapshots

