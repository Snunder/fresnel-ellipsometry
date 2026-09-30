"""Thickness-prefit utilities, model subsetting, and optional regularization controls."""

import numpy as np
import pandas as pd

from .models import MeasurementData, StackModel, thickness_parameter_name
from .dielectric import build_fit_parameters, build_initial_spline
from .fitting import fit_spline

def subset_stack_model(stack_model,mask):
    """Return a wavelength-subset copy of a stack model for local pre-fitting."""
    mask = np.asarray(mask,dtype=bool)
    known_eps = {i:np.asarray(eps)[mask].copy() for i,eps in stack_model.known_eps.items()}
    known_eps_e = {i:np.asarray(eps)[mask].copy() for i,eps in stack_model.known_eps_e.items()}
    thickness_defs = {name:dict(spec) for name,spec in stack_model.thickness_defs.items()}
    subset = StackModel(
        names=list(stack_model.names),
        unknown_index=stack_model.unknown_index,
        unknown_name=stack_model.unknown_name,
        known_eps=known_eps,
        thickness_defs=thickness_defs,
        wl_nm=np.asarray(stack_model.wl_nm)[mask].copy(),
        symmetry=stack_model.symmetry,
        known_eps_e=known_eps_e,
        thickness_anchors={name:dict(anchor) for name,anchor in getattr(stack_model,"thickness_anchors",{}).items()},
        dielectric_guard=dict(getattr(stack_model,"dielectric_guard",{}) or {}),
    )
    if hasattr(stack_model,"thickness_param_names"):
        subset.thickness_param_names = dict(stack_model.thickness_param_names)
    return subset

def subset_measurement(measurement,mask):
    """Return a wavelength-subset copy of a measurement for local pre-fitting."""
    mask = np.asarray(mask,dtype=bool)
    weights = np.asarray(measurement.spectral_weights,dtype=float)[mask].copy()
    weights /= np.mean(weights)
    return MeasurementData(
        wl_nm=np.asarray(measurement.wl_nm)[mask].copy(),
        AOIs=measurement.AOIs,
        psi_exp_deg={aoi:np.asarray(measurement.psi_exp_deg[aoi])[mask].copy() for aoi in measurement.AOIs},
        delta_exp_deg={aoi:np.asarray(measurement.delta_exp_deg[aoi])[mask].copy() for aoi in measurement.AOIs},
        psi_sigma_deg={aoi:np.asarray(measurement.psi_sigma_deg[aoi])[mask].copy() for aoi in measurement.AOIs},
        delta_sigma_deg={aoi:np.asarray(measurement.delta_sigma_deg[aoi])[mask].copy() for aoi in measurement.AOIs},
        spectral_weighting=measurement.spectral_weighting,
        spectral_weights=weights,)

def varying_thickness_names(stack_model):
    """Return finite layers whose thickness was supplied as a range."""
    return [name for name,spec in stack_model.thickness_defs.items() if spec["vary"]]

def seed_fitted_thicknesses(stack_model,fit_params,names=None):
    """Use fitted thicknesses as new starting values without changing their vary flags or bounds."""
    names = varying_thickness_names(stack_model) if names is None else list(names)
    seeded = {}
    for name in names:
        if name not in stack_model.thickness_defs:
            raise KeyError(f"Unknown finite layer {name!r}.")
        parameter_name = thickness_parameter_name(stack_model,name)
        if parameter_name not in fit_params:
            raise KeyError(f"Thickness parameter {parameter_name!r} is missing from the pre-fit result.")
        spec = stack_model.thickness_defs[name]
        value_nm = float(np.clip(fit_params[parameter_name].value,spec["min"],spec["max"]))
        spec["value"] = value_nm
        seeded[name] = value_nm
    return seeded

def set_thickness_anchors(stack_model,fit_result,names=None,relative_width=0.10,min_width_nm=5.0,stderr_multiplier=5.0,weight=1.0):
    """Anchor later joint fits to a thickness pre-fit without fixing the thickness.
    "scale" is deliberately broader than the local-fit covariance estimate because the
    pre-fit and full fit use the same spectrum.
    """
    names = varying_thickness_names(stack_model) if names is None else list(names)
    anchors = {}
    for name in names:
        par = fit_result.params[thickness_parameter_name(stack_model,name)]
        center = float(par.value)
        stderr = np.nan if par.stderr is None else float(par.stderr)
        candidates = [float(min_width_nm),abs(center)*float(relative_width)]
        if np.isfinite(stderr) and stderr > 0:
            candidates.append(float(stderr_multiplier)*stderr)
        scale = max(candidates)
        anchors[name] = {"value":center,"scale":scale,"weight":float(weight)}
    stack_model.thickness_anchors = anchors
    return {name:dict(anchor) for name,anchor in anchors.items()}

def clear_thickness_anchors(stack_model):
    """Disable soft thickness regularization while preserving current starting values and bounds."""
    stack_model.thickness_anchors = {}
    return stack_model

def set_dielectric_guard(stack_model,max_index=20.0,transition=1.0,weight=1.0,enabled=True):
    """Configure a sign-agnostic runaway-dielectric safety rail.

    "max_index" is the allowed magnitude of the complex refractive index. Since
    |N| = sqrt(|epsilon|), max_index=20 corresponds to |epsilon|=400. Negative
    epsilon_1 remains fully allowed. The guard is a penalty, not a clipped model.
    """
    max_index,transition,weight = float(max_index),float(transition),float(weight)
    if not np.isfinite(max_index) or max_index <= 0: raise ValueError("max_index must be positive and finite.")
    if not np.isfinite(transition) or transition <= 0: raise ValueError("transition must be positive and finite.")
    if not np.isfinite(weight) or weight < 0: raise ValueError("weight must be non-negative and finite.")
    stack_model.dielectric_guard = {"enabled":bool(enabled),"max_index":max_index,"transition":transition,"weight":weight}
    return dict(stack_model.dielectric_guard)

def _thickness_prefit_windows(wl_nm,window_nm=200.0,step_nm=50.0,min_points=20):
    """Generate overlapping windows from long to short wavelength."""
    wl_nm = np.asarray(wl_nm,dtype=float)
    wl_min,wl_max = float(np.min(wl_nm)),float(np.max(wl_nm))
    span = wl_max-wl_min
    width = min(float(window_nm),span)
    if width <= 0: return []
    if span <= width+1e-12: return [(wl_min,wl_max,np.ones_like(wl_nm,dtype=bool))]
    highs = list(np.arange(wl_max,wl_min+width-1e-12,-float(step_nm)))
    final_high = wl_min+width
    if len(highs) == 0 or abs(highs[-1]-final_high) > 0.5*float(step_nm): highs.append(final_high)
    windows = []
    for high in highs:
        low = max(wl_min,float(high)-width)
        high = min(wl_max,float(high))
        mask = (wl_nm >= low) & (wl_nm <= high)
        if np.count_nonzero(mask) >= int(min_points): windows.append((low,high,mask))
    return windows

def scan_thickness_prefit_windows(stack_model,measurement,bandwidth_nm,degree=3,window_nm=200.0,step_nm=50.0,free_nodes=2,max_nfev=5000,min_points=20,hc_ev_nm=1239.8419843320026,kk_points=601,eps_offset_bounds=(-np.inf,np.inf),eps2_coeff_bounds=(0.0,np.inf),eps_offset_parallel_init=3.0,eps_offset_perp_init=2.0,eps2_coeff_init=1e-3):
    """Fit a deliberately low-flexibility spline in moving windows and return the smoothest successful window."""
    if not stack_model.has_unknown: raise ValueError("Thickness pre-fitting with a local spline requires one unknown material.")
    thickness_names = varying_thickness_names(stack_model)
    if len(thickness_names) == 0: raise ValueError("No thickness range is available for pre-fitting.")
    bandwidth_nm = np.asarray(bandwidth_nm,dtype=float)
    windows = _thickness_prefit_windows(measurement.wl_nm,window_nm=window_nm,step_nm=step_nm,min_points=min_points)
    rows,objects = [],[]
    for window_index,(wl_low,wl_high,mask) in enumerate(windows):
        local_stack = subset_stack_model(stack_model,mask)
        local_measurement = subset_measurement(measurement,mask)
        local_bandwidth = bandwidth_nm[mask]
        E_local = hc_ev_nm/local_measurement.wl_nm
        kk_e_min = max(0.05,0.30*np.min(E_local))
        kk_e_max = 2.50*np.max(E_local)
        local_spline = build_initial_spline(local_measurement.wl_nm,local_bandwidth,degree=degree,initial_free_nodes=int(free_nodes),auto_nodes_min=int(free_nodes),auto_nodes_max=int(free_nodes),coarse_resolution_factor=12.0,kk_e_min=kk_e_min,kk_e_max=kk_e_max,kk_points=kk_points,hc_ev_nm=hc_ev_nm)
        local_params = build_fit_parameters(local_stack,local_spline,eps_offset_bounds,eps2_coeff_bounds,eps_offset_parallel_init,eps_offset_perp_init,eps2_coeff_init)
        local_result = fit_spline(local_params,local_spline,local_stack,local_measurement,max_nfev=max_nfev)
        sigma_residual = np.asarray(local_result.sigma_residual,dtype=float)
        rms_sigma = float(np.sqrt(np.mean(sigma_residual**2)))
        max_sigma = float(np.max(np.abs(sigma_residual)))
        row = {"window_index":window_index,"wl_low_nm":wl_low,"wl_high_nm":wl_high,"wl_center_nm":0.5*(wl_low+wl_high),"free_nodes":int(local_spline.n_free_nodes),"n_wavelengths":int(np.count_nonzero(mask)),"rms_sigma":rms_sigma,"max_sigma":max_sigma,"redchi":float(local_result.redchi),"success":bool(local_result.success)}
        for name in thickness_names:
            par = local_result.params[thickness_parameter_name(local_stack,name)]
            row[f"{name}_thickness_nm"] = float(par.value)
            row[f"{name}_stderr_nm"] = np.nan if par.stderr is None else float(par.stderr)
            spec = stack_model.thickness_defs[name]
            span = spec["max"]-spec["min"]
            row[f"{name}_at_bound"] = bool(np.isfinite(span) and span > 0 and min(par.value-spec["min"],spec["max"]-par.value) <= 0.01*span)
        rows.append(row)
        objects.append({"row":row,"result":local_result,"spline":local_spline,"stack_model":local_stack,"measurement":local_measurement})
    successful = [obj for obj in objects if obj["row"]["success"] and np.isfinite(obj["row"]["rms_sigma"])]
    if len(successful) == 0: raise RuntimeError("No local thickness pre-fit window converged successfully.")
    # A slightly lower RMS at a hard thickness bound is usually a poor initializer for
    # the later dielectric inversion. Prefer an interior solution whenever one exists.
    interior = [obj for obj in successful if not any(obj["row"][f"{name}_at_bound"] for name in thickness_names)]
    candidate_pool = interior if len(interior) else successful
    best = min(candidate_pool,key=lambda obj:obj["row"]["rms_sigma"])
    return {"history":pd.DataFrame(rows),"best":best,"thickness_names":thickness_names,"used_bound_fallback":len(interior)==0}

def automatic_thickness_prefit(stack_model,measurement,bandwidth_nm,degree=3,window_nm=200.0,step_nm=50.0,node_options=(2,3),target_rms_sigma=1.5,max_nfev=5000,min_points=20,hc_ev_nm=1239.8419843320026,kk_points=601,eps_offset_bounds=(-np.inf,np.inf),eps2_coeff_bounds=(0.0,np.inf),eps_offset_parallel_init=3.0,eps_offset_perp_init=2.0,eps2_coeff_init=1e-3):
    """Scan long-to-short wavelength windows with 2 then 3 free spline coefficients and choose the lowest-residual smooth region."""
    scans = []
    for free_nodes in node_options:
        scan = scan_thickness_prefit_windows(stack_model,measurement,bandwidth_nm,degree=degree,window_nm=window_nm,step_nm=step_nm,free_nodes=free_nodes,max_nfev=max_nfev,min_points=min_points,hc_ev_nm=hc_ev_nm,kk_points=kk_points,eps_offset_bounds=eps_offset_bounds,eps2_coeff_bounds=eps2_coeff_bounds,eps_offset_parallel_init=eps_offset_parallel_init,eps_offset_perp_init=eps_offset_perp_init,eps2_coeff_init=eps2_coeff_init)
        scans.append(scan)
        if scan["best"]["row"]["rms_sigma"] <= target_rms_sigma: break
    best_candidates = [scan["best"] for scan in scans]
    thickness_names = scans[0]["thickness_names"]
    interior_candidates = [obj for obj in best_candidates if not any(obj["row"][f"{name}_at_bound"] for name in thickness_names)]
    candidate_pool = interior_candidates if len(interior_candidates) else best_candidates
    best = min(candidate_pool,key=lambda obj:obj["row"]["rms_sigma"])
    history = pd.concat([scan["history"] for scan in scans],ignore_index=True)
    return {"history":history,"best":best,"thickness_names":thickness_names,"node_counts_tested":[int(scan["best"]["row"]["free_nodes"]) for scan in scans],"used_bound_fallback":len(interior_candidates)==0}



def print_thickness_prefit_summary(prefit,label=None,precision=3):
    """Print a concise summary of the selected local thickness pre-fit."""
    if prefit is None or "best" not in prefit:
        raise ValueError("prefit must be the result returned by automatic_thickness_prefit().")
    row = prefit["best"]["row"]
    title = "Thickness pre-fit" if label is None else f"Thickness pre-fit — {label}"
    print(title)
    print(
        f"  selected window: {row['wl_low_nm']:.1f}–{row['wl_high_nm']:.1f} nm | "
        f"{int(row['free_nodes'])} spline coefficients | RMS={row['rms_sigma']:.3f} sigma"
    )
    for name in prefit.get("thickness_names", []):
        value = float(row[f"{name}_thickness_nm"])
        stderr = float(row[f"{name}_stderr_nm"])
        stderr_text = "" if not np.isfinite(stderr) else f" ± {stderr:.{int(precision)}f} nm"
        bound_text = " [near input bound]" if bool(row[f"{name}_at_bound"]) else ""
        print(f"  {name}: {value:.{int(precision)}f} nm{stderr_text}{bound_text}")
    if bool(prefit.get("used_bound_fallback", False)):
        print("  note: no interior pre-fit solution was available; the best bound-adjacent solution was used.")
    return row
