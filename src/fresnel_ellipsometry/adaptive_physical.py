"""Adaptive physical-dispersion model selection for a single sample."""

import numpy as np
import pandas as pd

from .dielectric import (
    _canonical_physical_kind, add_physical_term, build_physical_parameters,
)
from .fitting import (
    _fit_measurement_chisqr, _fit_measurement_redchi, fit_bic, fit_physical,
    residual_search_metrics, residual_snapshot,
)

def _rank_physical_centers(params,model,stack_model,measurement,bandwidth_nm,n_centers=3,residual_smooth_factor=0.5):
    """Return separated residual peaks used as nonlinear line-shape seeds."""
    search = residual_search_metrics(params,model,stack_model,measurement,bandwidth_nm,residual_smooth_factor=residual_smooth_factor)
    wl = np.asarray(measurement.wl_nm,dtype=float)
    score = np.maximum(search["regional_rms"],search["individual_max"])
    local_bandwidth = np.asarray(bandwidth_nm,dtype=float)
    centers = []
    for idx in np.argsort(score)[::-1]:
        candidate = float(wl[idx])
        separation = max(2.0*float(local_bandwidth[idx]),0.04*float(np.ptp(wl)))
        if all(abs(candidate-existing) >= separation for existing in centers): centers.append(candidate)
        if len(centers) >= int(n_centers): break
    if not centers: centers = [float(np.median(wl))]
    return centers,search

def _physical_seeds(kind,center_wavelengths_nm,model):
    """Small deterministic seed bank for hierarchical physical-family screening."""
    kind = _canonical_physical_kind(kind)
    E_meas = model.hc_ev_nm/np.asarray(model.wl_nm,dtype=float)
    e_min,e_max = float(np.min(E_meas)),float(np.max(E_meas))
    e_span = max(e_max-e_min,0.1)
    wl_min,wl_max = float(np.min(model.wl_nm)),float(np.max(model.wl_nm))
    if kind == "cauchy":
        return [("weak curvature",{"B":0.003,"C":0.0001}),("medium curvature",{"B":0.02,"C":0.002}),("strong curvature",{"B":0.08,"C":0.01})]
    if kind == "sellmeier_uv":
        return [(f"UV pole={fraction:.2f}lambda_min",{"strength":1.0,"lambda0_nm":max(1.0,fraction*wl_min)}) for fraction in (0.25,0.55,0.85)]
    if kind == "sellmeier_ir":
        return [(f"IR pole={factor:.1f}lambda_max",{"strength":1.0,"lambda0_nm":factor*wl_max}) for factor in (1.2,2.0,5.0)]
    if kind == "drude":
        return [(f"gamma={gamma:.3g}eV",{"amp":max(0.1,e_min**2),"gamma":gamma}) for gamma in (max(0.005,0.02*e_min),max(0.01,0.10*e_min),max(0.03,0.50*e_min))]
    if kind in ("lorentz","harmonic","gaussian"):
        return [(f"center={wl:.1f}nm",{"amp":5.0,"e0":model.hc_ev_nm/wl,"gamma":max(0.03,0.12*e_span)}) for wl in center_wavelengths_nm]
    if kind in ("tauc_lorentz","cody_lorentz"):
        seeds = []
        for wl in center_wavelengths_nm:
            e0 = model.hc_ev_nm/wl
            eg = max(0.0,e0-0.35*e_span)
            seed = {"amp":20.0,"eg":eg,"de":max(0.02,e0-eg),"gamma":max(0.03,0.15*e_span)}
            if kind == "cody_lorentz": seed["ep"] = max(0.03,0.12*e_span)
            seeds.append((f"center={wl:.1f}nm",seed))
        return seeds
    raise ValueError(f"Unsupported physical term {kind!r}.")

def adaptive_fit_physical(params,model,stack_model,measurement,bandwidth_nm,candidate_types=("cauchy","sellmeier_uv","sellmeier_ir","drude","lorentz","harmonic","gaussian","tauc_lorentz","cody_lorentz"),max_terms=5,max_nfev=100000,trial_max_nfev=600,screen_max_nfev=180,n_family_finalists=4,bic_improvement=6.0,residual_threshold=1.5,individual_residual_threshold=4.0,residual_smooth_factor=0.5,n_centers=3,verbose=False,show_progress=True):
    """Select dispersion families sequentially using hierarchical multistart BIC.
    Every family receives one short screen. Only the best few families receive
    their remaining starts. Accepted terms and their fitted values initialize the
    next iteration.
    """
    candidate_types = tuple(_canonical_physical_kind(kind) for kind in candidate_types)
    allowed = {"cauchy","sellmeier_uv","sellmeier_ir","drude","lorentz","harmonic","gaussian","tauc_lorentz","cody_lorentz"}
    invalid = sorted(set(candidate_types)-allowed)
    if invalid: raise ValueError(f"Unsupported physical candidate type(s): {invalid}.")
    if not candidate_types: raise ValueError("candidate_types may not be empty.")
    if show_progress: print(f"Physical iteration 0/{int(max_terms)}",end="\r",flush=True)
    fit_result = fit_physical(params,model,stack_model,measurement,max_nfev=max_nfev)
    history = [{"iteration":0,"mode":"baseline","added_type":"none","model":model.description,"n_components":0,"chisqr":_fit_measurement_chisqr(fit_result),"redchi":_fit_measurement_redchi(fit_result),"bic":fit_bic(fit_result),"bic_gain":np.nan,"accepted":True}]
    snapshots = [residual_snapshot(fit_result,model,stack_model,measurement,0)]
    for iteration in range(1,int(max_terms)+1):
        if show_progress: print(f"Physical iteration {iteration}/{int(max_terms)}",end="\r",flush=True)
        centers,search = _rank_physical_centers(fit_result.params,model,stack_model,measurement,bandwidth_nm,n_centers=n_centers,residual_smooth_factor=residual_smooth_factor)
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
            proposal_params = build_physical_parameters(stack_model,proposal_model,previous_result=fit_result,new_term_seed=seed)
            trial = fit_physical(proposal_params,proposal_model,stack_model,measurement,max_nfev=min(int(screen_max_nfev),int(trial_max_nfev)))
            gain = current_bic-fit_bic(trial)
            record = {"kind":kind,"model":proposal_model,"seeds":seeds,"best_seed":seed_label,"best_result":trial,"bic_gain":gain}
            family_records.append(record)
        if not family_records:
            if verbose: print("Stopping: no candidate families remain.")
            break
        finalists = sorted(family_records,key=lambda item:item["bic_gain"],reverse=True)[:max(1,int(n_family_finalists))]
        for record in finalists:
            for seed_label,seed in record["seeds"][1:]:
                proposal_params = build_physical_parameters(stack_model,record["model"],previous_result=fit_result,new_term_seed=seed)
                trial = fit_physical(proposal_params,record["model"],stack_model,measurement,max_nfev=trial_max_nfev)
                gain = current_bic-fit_bic(trial)
                if np.isfinite(_fit_measurement_chisqr(trial)) and gain > record["bic_gain"]:
                    record.update({"best_seed":seed_label,"best_result":trial,"bic_gain":gain})
        chosen = max(family_records,key=lambda proposal:proposal["bic_gain"])
        if not np.isfinite(_fit_measurement_chisqr(chosen["best_result"])) or chosen["bic_gain"] < bic_improvement:
            if verbose: print(f'Stopping: best candidate {chosen["kind"]} improved BIC by {chosen["bic_gain"]:.2f}.')
            break
        converged = fit_physical(chosen["best_result"].params,chosen["model"],stack_model,measurement,max_nfev=max_nfev)
        final_gain = current_bic-fit_bic(converged)
        accepted = bool(converged.success and np.isfinite(_fit_measurement_chisqr(converged)) and final_gain >= bic_improvement)
        component_count = chosen["model"].n_terms+int(chosen["model"].background != "constant")
        history.append({"iteration":iteration,"mode":"physical","added_type":chosen["kind"],"model":chosen["model"].description,"n_components":component_count,"chisqr":_fit_measurement_chisqr(converged),"redchi":_fit_measurement_redchi(converged),"bic":fit_bic(converged),"bic_gain":final_gain,"accepted":accepted})
        if verbose: print(f'Selected {chosen["kind"]}; ΔBIC={final_gain:.2f}.')
        if not accepted: break
        model,fit_result = chosen["model"],converged
        snapshots.append(residual_snapshot(fit_result,model,stack_model,measurement,iteration))
    if show_progress: print()
    return fit_result,model,pd.DataFrame(history),snapshots

