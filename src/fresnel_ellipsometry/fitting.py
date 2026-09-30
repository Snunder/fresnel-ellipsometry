"""Single-sample residuals, objective functions, and optimizers."""

from types import SimpleNamespace
import numpy as np
from lmfit import minimize
from scipy.ndimage import gaussian_filter1d

from .models import PhysicalModel, SplineModel, thickness_parameter_name
from .dielectric import dielectric_from_model
from .fresnel import fresnel_reflection_all, wrap_delta_deg

def ellipsometry_residuals(params,optical_model,stack_model,measurement):
    """Return unweighted Psi/Delta residuals in degrees and measurement-sigma units."""
    psi_deg,delta_deg,psi_sigma,delta_sigma = {},{},{},{}
    responses = fresnel_reflection_all(params,measurement.AOIs,optical_model,stack_model)
    for aoi in measurement.AOIs:
        result = responses[float(aoi)]
        psi_deg[aoi] = result["psi"]-measurement.psi_exp_deg[aoi]
        delta_deg[aoi] = wrap_delta_deg(result["delta"]-measurement.delta_exp_deg[aoi])
        psi_sigma[aoi] = psi_deg[aoi]/measurement.psi_sigma_deg[aoi]
        delta_sigma[aoi] = delta_deg[aoi]/measurement.delta_sigma_deg[aoi]
    return {"psi_deg":psi_deg,"delta_deg":delta_deg,"psi_sigma":psi_sigma,"delta_sigma":delta_sigma}

def _stack_sigma_residual(residual,measurement,weighted=False):
    scale = np.sqrt(np.asarray(measurement.spectral_weights,dtype=float)) if weighted else 1.0
    return np.concatenate([
        component*scale
        for aoi in measurement.AOIs
        for component in (residual["psi_sigma"][aoi],residual["delta_sigma"][aoi])
    ])

def _fit_residual(params,optical_model,stack_model,measurement):
    residual = ellipsometry_residuals(params,optical_model,stack_model,measurement)
    return _stack_sigma_residual(residual,measurement,weighted=True)

def _thickness_anchor_residual(params,stack_model,measurement):
    """Return optional soft-regularization residuals for poorly identifiable thicknesses.

    Anchors are numerical regularizers, not additional experimental observations.  Each
    anchor is scaled by sqrt(N_data) so its ``weight`` is approximately comparable to a
    change of that size in the mean squared normalized ellipsometry residual.
    """
    anchors = getattr(stack_model,"thickness_anchors",{}) or {}
    if len(anchors) == 0:
        return np.array([],dtype=float)
    n_data = max(1,2*len(measurement.AOIs)*len(measurement.wl_nm))
    values = []
    for name,anchor in anchors.items():
        parameter_name = thickness_parameter_name(stack_model,name)
        if parameter_name not in params:
            continue
        center = float(anchor["value"])
        scale = max(float(anchor["scale"]),1e-12)
        weight = max(float(anchor.get("weight",1.0)),0.0)
        if weight == 0:
            continue
        values.append(np.sqrt(weight*n_data)*(float(params[parameter_name].value)-center)/scale)
    return np.asarray(values,dtype=float)

def _dielectric_guard_residual(params,optical_model,stack_model,measurement):
    """Softly penalize only implausibly large |N| while allowing epsilon_1 < 0."""
    guard = getattr(stack_model,"dielectric_guard",{}) or {}
    if not guard or not bool(guard.get("enabled",True)) or optical_model is None or not stack_model.has_unknown:
        return np.array([],dtype=float)
    max_index = float(guard.get("max_index",20.0))
    transition = max(float(guard.get("transition",1.0)),1e-12)
    weight = max(float(guard.get("weight",1.0)),0.0)
    if not np.isfinite(max_index) or max_index <= 0 or weight == 0:
        return np.array([],dtype=float)
    eps_o,eps_e = dielectric_from_model(params,optical_model,stack_model)
    arrays = [np.sqrt(np.abs(np.asarray(eps_o,dtype=complex)))]
    if not stack_model.is_isotropic:
        arrays.append(np.sqrt(np.abs(np.asarray(eps_e,dtype=complex))))
    index_magnitude = np.concatenate([np.ravel(v) for v in arrays])
    excess = np.maximum(index_magnitude-max_index,0.0)/transition
    # scipy.optimize.least_squares requires the residual vector to keep exactly
    # the same length for every parameter evaluation.  Therefore the dielectric
    # guard must return one residual per evaluated dielectric point even when no
    # point violates the limit; in that case the entries are simply all zero.
    n_data = max(1,2*len(measurement.AOIs)*len(measurement.wl_nm))
    return np.sqrt(weight*n_data/max(len(excess),1))*excess

def _regularized_fit_residual(params,optical_model,stack_model,measurement):
    data_residual = _fit_residual(params,optical_model,stack_model,measurement)
    anchor_residual = _thickness_anchor_residual(params,stack_model,measurement)
    dielectric_residual = _dielectric_guard_residual(params,optical_model,stack_model,measurement)
    pieces = [data_residual]
    if anchor_residual.size: pieces.append(anchor_residual)
    if dielectric_residual.size: pieces.append(dielectric_residual)
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

def _annotate_data_fit(result,optical_model,stack_model,measurement):
    residual = ellipsometry_residuals(result.params,optical_model,stack_model,measurement)
    weighted = _stack_sigma_residual(residual,measurement,weighted=True)
    unweighted = _stack_sigma_residual(residual,measurement,weighted=False)
    data_chisqr = float(np.sum(weighted**2))
    data_ndata = int(weighted.size)
    data_dof = max(data_ndata-int(result.nvarys),1)
    result.data_residual = weighted
    result.sigma_residual = unweighted
    result.data_chisqr = data_chisqr
    result.data_ndata = data_ndata
    result.data_redchi = data_chisqr/data_dof
    result.sigma_rms = float(np.sqrt(np.mean(unweighted**2)))
    result.spectral_weighting = measurement.spectral_weighting
    result.regularization_chisqr = max(0.0,float(result.chisqr)-data_chisqr)
    return result

def _fit_measurement_chisqr(result):
    return float(getattr(result,"data_chisqr",result.chisqr))

def _fit_measurement_redchi(result):
    return float(getattr(result,"data_redchi",result.redchi))

def _fit_measurement_ndata(result):
    return int(getattr(result,"data_ndata",result.ndata))

def fit_thicknesses(params,stack_model,measurement,max_nfev=3000,profile_single_thickness=True,profile_points=241,profile_max_nm=None,profile_finalists=5):
    """Fit varying thicknesses while keeping all loaded optical constants fixed.

    When exactly one thickness varies, the default behavior first screens a 1-D
    thickness profile before local least-squares refinement.  This avoids the common
    failure mode where a broad interval such as ``[0, inf]`` starts near 100 nm and the
    optimizer converges to the wrong interference-order minimum.
    """
    if stack_model.has_unknown:
        raise ValueError("fit_thicknesses() is only used when all material optical constants are known.")
    varying = [name for name,par in params.items() if par.vary]
    if not varying:
        residual = _fit_residual(params,None,stack_model,measurement)
        unweighted = _stack_sigma_residual(
            ellipsometry_residuals(params,None,stack_model,measurement),
            measurement,
            weighted=False,
        )
        chisqr = float(np.sum(residual**2))
        ndata = int(residual.size)
        return SimpleNamespace(
            params=params,residual=residual,data_residual=residual,sigma_residual=unweighted,
            chisqr=chisqr,data_chisqr=chisqr,redchi=chisqr/max(ndata,1),
            data_redchi=chisqr/max(ndata,1),ndata=ndata,data_ndata=ndata,nvarys=0,
            sigma_rms=float(np.sqrt(np.mean(unweighted**2))),success=True,
            message="No varying thickness parameters.",covar=None,var_names=[],
            spectral_weighting=measurement.spectral_weighting,regularization_chisqr=0.0,
        )

    # A one-dimensional profile is inexpensive and much more reliable than trusting a
    # single local start for a periodic thin-film interference objective.
    if profile_single_thickness and len(varying) == 1:
        parameter_name = varying[0]
        par = params[parameter_name]
        lo = float(par.min) if np.isfinite(par.min) else 0.0
        if np.isfinite(par.max):
            hi = float(par.max)
        else:
            if profile_max_nm is None:
                profile_max_nm = max(1000.0,2.0*float(np.max(measurement.wl_nm)))
            hi = max(lo+1.0,float(profile_max_nm))

        grid = np.linspace(lo,hi,max(int(profile_points),25))
        scores = np.empty(grid.size,dtype=float)
        trial = params.copy()
        for i,value in enumerate(grid):
            trial[parameter_name].set(value=float(value))
            residual = _fit_residual(trial,None,stack_model,measurement)
            scores[i] = float(np.dot(residual,residual))

        minima = []
        if grid.size >= 3:
            minima.extend(i for i in range(1,grid.size-1) if scores[i] <= scores[i-1] and scores[i] <= scores[i+1])
        minima.extend([0,grid.size-1,int(np.argmin(scores))])
        minima = sorted(set(minima),key=lambda i:scores[i])
        finalist_indices = minima[:max(1,int(profile_finalists))]

        # Always retain the user's original seed as an additional candidate.
        start_values = [float(grid[i]) for i in finalist_indices]
        original_value = float(par.value)
        if not any(np.isclose(original_value,value,rtol=0.0,atol=1e-12) for value in start_values):
            start_values.append(original_value)

        best = None
        best_chisqr = np.inf
        for value in start_values:
            candidate = params.copy()
            candidate[parameter_name].set(value=float(np.clip(value,par.min,par.max)))
            result = minimize(
                lambda p:_fit_residual(p,None,stack_model,measurement),
                candidate,method="least_squares",max_nfev=max_nfev,scale_covar=False,
            )
            annotated = _annotate_data_fit(result,None,stack_model,measurement)
            if annotated.data_chisqr < best_chisqr:
                best,best_chisqr = annotated,float(annotated.data_chisqr)

        best.thickness_profile = {
            "parameter":parameter_name,
            "grid_nm":grid,
            "chisqr":scores,
            "start_values_nm":np.asarray(start_values,dtype=float),
            "search_max_nm":float(hi),
        }
        return best

    result = minimize(
        lambda p:_fit_residual(p,None,stack_model,measurement),
        params,method="least_squares",max_nfev=max_nfev,scale_covar=False,
    )
    return _annotate_data_fit(result,None,stack_model,measurement)

def fit_spline(params, spline, stack_model, measurement, max_nfev=3000):
    """Fit spline coefficients and free thicknesses with optional numerical regularization."""
    if not stack_model.has_unknown: raise ValueError("All optical constants are known; use fit_thicknesses() instead.")
    if not isinstance(spline,SplineModel): raise TypeError("fit_spline() requires a SplineModel.")
    result = minimize(lambda p:_regularized_fit_residual(p,spline,stack_model,measurement),params,method="least_squares",max_nfev=max_nfev,scale_covar=False)
    return _annotate_data_fit(result,spline,stack_model,measurement)

def fit_physical(params,model,stack_model,measurement,max_nfev=3000):
    """Fit a physical dielectric model and free thicknesses with optional numerical regularization."""
    if not stack_model.has_unknown: raise ValueError("All optical constants are known; use fit_thicknesses() instead.")
    if not isinstance(model,PhysicalModel): raise TypeError("model must be a PhysicalModel.")
    result = minimize(lambda p:_regularized_fit_residual(p,model,stack_model,measurement),params,method="least_squares",max_nfev=max_nfev,scale_covar=False)
    return _annotate_data_fit(result,model,stack_model,measurement)

def wavelength_residual(params,spline,stack_model,measurement,residual=None):
    """Return the local RMS sigma-normalized Psi/Delta residual across all AOIs."""
    if residual is None:
        residual = ellipsometry_residuals(params,spline,stack_model,measurement)
    residual_squared = [r**2 for aoi in measurement.AOIs for r in (residual["psi_sigma"][aoi],residual["delta_sigma"][aoi])]
    return np.sqrt(np.mean(np.stack(residual_squared),axis=0))

def _normalized_gaussian_smooth(values,sigma_samples):
    """Gaussian local average using only measured points at the spectral boundaries."""
    values = np.asarray(values,dtype=float)
    if sigma_samples <= 0: return values.copy()
    numerator = gaussian_filter1d(values,sigma_samples,mode="constant",cval=0.0)
    denominator = gaussian_filter1d(np.ones_like(values),sigma_samples,mode="constant",cval=0.0)
    return numerator/np.maximum(denominator,1e-15)

def residual_search_metrics(params,spline,stack_model,measurement,bandwidth_nm,residual_smooth_factor=1.0):
    """Return regional, raw-RMS and worst-individual normalized residual curves."""
    wl_nm = np.asarray(measurement.wl_nm,dtype=float)
    bandwidth_nm = np.asarray(bandwidth_nm,dtype=float)
    wl_step_nm = np.median(np.diff(np.sort(wl_nm))) if len(wl_nm) > 1 else 1.0
    residual = ellipsometry_residuals(params,spline,stack_model,measurement)
    raw_rms = wavelength_residual(params,spline,stack_model,measurement,residual=residual)
    sigma_samples = residual_smooth_factor*np.median(bandwidth_nm)/max(wl_step_nm,1e-12)
    regional_rms = _normalized_gaussian_smooth(raw_rms,max(sigma_samples,0.0))
    individual_stack = np.stack([np.abs(r) for aoi in measurement.AOIs for r in (residual["psi_sigma"][aoi],residual["delta_sigma"][aoi])])
    individual_max = np.max(individual_stack,axis=0)
    return {"raw_rms":raw_rms,"regional_rms":regional_rms,"individual_max":individual_max,"max_raw_rms":float(np.max(raw_rms)),"max_regional_rms":float(np.max(regional_rms)),"max_individual_residual":float(np.max(individual_max))}

def residual_snapshot(result,optical_model,stack_model,measurement,iteration):
    residual = ellipsometry_residuals(result.params,optical_model,stack_model,measurement)
    return {
        "iteration": int(iteration),
        "psi_residual_deg": {aoi: (values).copy() for aoi,values in residual["psi_deg"].items()},
        "delta_residual_deg": {aoi: (values).copy() for aoi,values in residual["delta_deg"].items()},
    }

def fit_bic(result):
    """BIC from weighted ellipsometry residuals only; numerical regularizers are excluded."""
    return _fit_measurement_chisqr(result)+result.nvarys*np.log(_fit_measurement_ndata(result))

