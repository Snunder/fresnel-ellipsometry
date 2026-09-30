"""Core joint multi-sample fitting with a shared dielectric function and independent thicknesses."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize

from .models import PhysicalModel, configure_multifit_thickness_parameters, thickness_parameter_name
from .dielectric import build_fit_parameters, build_physical_parameters, dielectric_from_model
from .fresnel import _get_forward_cache, _multiflake_residual_core_numba
from .fitting import _stack_sigma_residual, _thickness_anchor_residual, ellipsometry_residuals

def _prepare_batched_multifit_backend(datasets):
    """Prepare a compiled joint-residual backend when flakes share the same optical grid.

    The fast path is exact: it is used only when AOIs, wavelengths, stack ordering,
    symmetry, unknown-layer position and every known dielectric array are identical.
    Thicknesses may differ freely between flakes. Otherwise the ordinary per-flake
    residual path is used.
    """
    datasets = list(datasets)
    if _multiflake_residual_core_numba is None or len(datasets) < 2:
        return None
    first = datasets[0]
    reference = first.stack_model
    measurement = first.measurement
    aois = tuple(float(aoi) for aoi in measurement.AOIs)
    wl_nm = np.asarray(measurement.wl_nm,dtype=float)
    known_keys = tuple(sorted(reference.known_eps))
    for dataset in datasets[1:]:
        stack_model = dataset.stack_model
        local = dataset.measurement
        if tuple(float(aoi) for aoi in local.AOIs) != aois:
            return None
        if not np.array_equal(np.asarray(local.wl_nm,dtype=float),wl_nm):
            return None
        if tuple(stack_model.names) != tuple(reference.names) or stack_model.symmetry != reference.symmetry:
            return None
        if stack_model.unknown_index != reference.unknown_index:
            return None
        if tuple(sorted(stack_model.known_eps)) != known_keys:
            return None
        for i in known_keys:
            if not np.array_equal(np.asarray(stack_model.known_eps[i]),np.asarray(reference.known_eps[i])):
                return None
            if not np.array_equal(np.asarray(stack_model.known_eps_e.get(i,stack_model.known_eps[i])),
                                  np.asarray(reference.known_eps_e.get(i,reference.known_eps[i]))):
                return None

    cache = _get_forward_cache(reference,aois)
    n_flakes,n_aoi,n_wl = len(datasets),len(aois),len(wl_nm)
    psi_exp = np.empty((n_flakes,n_aoi,n_wl),dtype=float)
    delta_exp = np.empty_like(psi_exp)
    psi_sigma = np.empty_like(psi_exp)
    delta_sigma = np.empty_like(psi_exp)
    sqrt_weights = np.empty((n_flakes,n_wl),dtype=float)
    for f,dataset in enumerate(datasets):
        local = dataset.measurement
        sqrt_weights[f] = np.sqrt(np.asarray(local.spectral_weights,dtype=float))
        for a,aoi in enumerate(aois):
            psi_exp[f,a] = local.psi_exp_deg[aoi]
            delta_exp[f,a] = local.delta_exp_deg[aoi]
            psi_sigma[f,a] = local.psi_sigma_deg[aoi]
            delta_sigma[f,a] = local.delta_sigma_deg[aoi]

    return SimpleNamespace(
        datasets=datasets,reference_stack=reference,aois=aois,wl_nm=wl_nm,cache=cache,
        psi_exp=psi_exp,delta_exp=delta_exp,psi_sigma=psi_sigma,delta_sigma=delta_sigma,
        sqrt_weights=sqrt_weights,
    )

def _batched_multifit_thickness_array(params,backend):
    n_flakes = len(backend.datasets)
    n_layers = len(backend.reference_stack.names)
    d_all_nm = np.zeros((n_flakes,n_layers),dtype=float)
    for f,dataset in enumerate(backend.datasets):
        for i,name in enumerate(dataset.stack_model.names[1:-1],1):
            d_all_nm[f,i] = params[thickness_parameter_name(dataset.stack_model,name)].value
    return d_all_nm

def _batched_multifit_data_residual(params,optical_model,backend):
    reference = backend.reference_stack
    if reference.has_unknown:
        eps_o,eps_e = dielectric_from_model(params,optical_model,reference,wl_eval_nm=backend.wl_nm)
        eps_o = np.asarray(eps_o,dtype=np.complex128)
        eps_e = np.asarray(eps_e,dtype=np.complex128)
        unknown_index = int(reference.unknown_index)
    else:
        eps_o = np.ones(len(backend.wl_nm),dtype=np.complex128)
        eps_e = eps_o
        unknown_index = -1
    d_all_nm = _batched_multifit_thickness_array(params,backend)
    cache = backend.cache
    residual = _multiflake_residual_core_numba(
        cache["q2"],cache["qz_s"],cache["qz_p"],cache["y_s"],cache["y_p"],cache["k0"][0],
        eps_o,eps_e,d_all_nm,unknown_index,
        backend.psi_exp,backend.delta_exp,backend.psi_sigma,backend.delta_sigma,backend.sqrt_weights,
    )
    return residual,eps_o,eps_e

def _dielectric_guard_from_epsilon(eps_o,eps_e,datasets):
    reference = datasets[0].stack_model
    guard = getattr(reference,"dielectric_guard",{}) or {}
    if not reference.has_unknown or not guard.get("enabled",False) or float(guard.get("weight",0.0)) <= 0:
        return np.array([],dtype=float)
    arrays = [np.sqrt(np.abs(np.asarray(eps_o,dtype=complex)))]
    if reference.is_uniaxial:
        arrays.append(np.sqrt(np.abs(np.asarray(eps_e,dtype=complex))))
    magnitude = np.concatenate(arrays)
    max_index = float(guard["max_index"])
    transition = float(guard["transition"])
    weight = float(guard["weight"])
    excess = np.maximum(magnitude-max_index,0.0)/transition
    n_data = max(1,sum(2*len(dataset.measurement.AOIs)*len(dataset.measurement.wl_nm) for dataset in datasets))
    return np.sqrt(weight*n_data/max(len(excess),1))*excess

def _batched_multifit_regularized_residual(params,optical_model,backend):
    data_residual,eps_o,eps_e = _batched_multifit_data_residual(params,optical_model,backend)
    pieces = [data_residual]
    anchors = _multi_anchor_residual(params,backend.datasets)
    guard = _dielectric_guard_from_epsilon(eps_o,eps_e,backend.datasets)
    if anchors.size:
        pieces.append(anchors)
    if guard.size:
        pieces.append(guard)
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

def multifit_backend_info(datasets):
    """Describe whether a set of flakes can use the batched Numba residual backend."""
    backend = _prepare_batched_multifit_backend(datasets)
    if backend is None:
        return ""
    return f"batched-numba ({len(backend.datasets)} flakes x {len(backend.aois)} AOIs x {len(backend.wl_nm)} wavelengths)"

def _validate_known_multifit_datasets(datasets):
    """Validate several fully known stacks used only for independent thickness parameters."""
    datasets = list(datasets)
    if len(datasets) < 2:
        raise ValueError("A multi-dataset thickness fit requires at least two datasets.")
    if any(dataset.stack_model.has_unknown for dataset in datasets):
        raise ValueError("Every dataset must have optical constants for all materials.")
    symmetries = {dataset.stack_model.symmetry for dataset in datasets}
    stacks = {tuple(dataset.stack_model.names) for dataset in datasets}
    if len(symmetries) != 1:
        raise ValueError("Every dataset must use the same symmetry.")
    if len(stacks) != 1:
        raise ValueError("Every dataset must use the same layer ordering.")
    return datasets

def build_multi_thickness_parameters(datasets):
    """Create independent finite-layer thickness parameters for several fully known datasets."""
    datasets = configure_multifit_thickness_parameters(_validate_known_multifit_datasets(datasets))
    p = Parameters()
    for dataset in datasets:
        for name,spec in dataset.stack_model.thickness_defs.items():
            pname = thickness_parameter_name(dataset.stack_model,name)
            if spec["vary"]:
                p.add(pname,value=spec["value"],min=spec["min"],max=spec["max"],vary=True)
            else:
                p.add(pname,value=spec["value"],vary=False)
    return p

def _multi_thickness_data_residual(params,datasets,weighted=True):
    pieces = []
    for dataset in _validate_known_multifit_datasets(datasets):
        residual = ellipsometry_residuals(params,None,dataset.stack_model,dataset.measurement)
        pieces.append(_stack_sigma_residual(residual,dataset.measurement,weighted=weighted))
    return np.concatenate(pieces)

def _annotate_multi_thickness_fit(result,datasets):
    datasets = _validate_known_multifit_datasets(datasets)
    weighted = _multi_thickness_data_residual(result.params,datasets,weighted=True)
    unweighted = _multi_thickness_data_residual(result.params,datasets,weighted=False)
    data_chisqr = float(np.sum(weighted**2))
    data_ndata = int(weighted.size)
    data_dof = max(data_ndata-int(result.nvarys),1)
    result.data_residual = weighted
    result.sigma_residual = unweighted
    result.data_chisqr = data_chisqr
    result.data_ndata = data_ndata
    result.data_redchi = data_chisqr/data_dof
    result.sigma_rms = float(np.sqrt(np.mean(unweighted**2)))
    result.regularization_chisqr = 0.0
    result.spectral_weighting = tuple(sorted({dataset.measurement.spectral_weighting for dataset in datasets}))
    metrics = {}
    for dataset in datasets:
        residual = ellipsometry_residuals(result.params,None,dataset.stack_model,dataset.measurement)
        sigma = _stack_sigma_residual(residual,dataset.measurement,weighted=False)
        psi_deg = np.concatenate([residual["psi_deg"][aoi] for aoi in dataset.measurement.AOIs])
        delta_deg = np.concatenate([residual["delta_deg"][aoi] for aoi in dataset.measurement.AOIs])
        metrics[dataset.label] = {
            "rms_sigma":float(np.sqrt(np.mean(sigma**2))),
            "max_sigma":float(np.max(np.abs(sigma))),
            "psi_rms_deg":float(np.sqrt(np.mean(psi_deg**2))),
            "delta_rms_deg":float(np.sqrt(np.mean(delta_deg**2))),
        }
    result.dataset_metrics = metrics
    return result

def fit_thicknesses_multi(params,datasets,max_nfev=3000):
    """Joint optimizer call for several fully known stacks; only their thicknesses vary."""
    datasets = configure_multifit_thickness_parameters(_validate_known_multifit_datasets(datasets))
    backend = _prepare_batched_multifit_backend(datasets)
    varying = [name for name,par in params.items() if par.vary]
    def data_residual(p):
        return _batched_multifit_data_residual(p,None,backend)[0] if backend is not None else _multi_thickness_data_residual(p,datasets,weighted=True)
    if not varying:
        weighted = data_residual(params)
        unweighted = _multi_thickness_data_residual(params,datasets,weighted=False)
        chisqr = float(np.sum(weighted**2))
        ndata = int(weighted.size)
        result = SimpleNamespace(
            params=params,residual=weighted,chisqr=chisqr,redchi=chisqr/max(ndata,1),
            ndata=ndata,nvarys=0,success=True,message="No varying thickness parameters.",
            covar=None,var_names=[],
        )
        return _annotate_multi_thickness_fit(result,datasets)
    result = minimize(
        data_residual,
        params,method="least_squares",max_nfev=max_nfev,scale_covar=False,
    )
    result.fast_multifit_backend = backend is not None
    return _annotate_multi_thickness_fit(result,datasets)

def _validate_multifit_datasets(datasets):
    datasets = list(datasets)
    if len(datasets) < 2:
        raise ValueError("A multi-flake fit requires at least two datasets.")
    unknown_names = {dataset.stack_model.unknown_name for dataset in datasets}
    symmetries = {dataset.stack_model.symmetry for dataset in datasets}
    stacks = {tuple(dataset.stack_model.names) for dataset in datasets}
    if len(unknown_names) != 1 or None in unknown_names:
        raise ValueError("Every flake must reconstruct the same one unknown material.")
    if len(symmetries) != 1:
        raise ValueError("Every flake must use the same symmetry.")
    if len(stacks) != 1:
        raise ValueError("Every flake must use the same layer ordering.")
    return datasets

def build_multi_fit_parameters(datasets,spline,eps_offset_bounds=(-np.inf,np.inf),eps2_coeff_bounds=(0.0,np.inf),eps_offset_parallel_init=4.0,eps_offset_perp_init=2.5,eps2_coeff_init=1e-3,previous_result=None,old_spline=None):
    """One shared spline dielectric parameterization plus independent thicknesses for every flake."""
    datasets = configure_multifit_thickness_parameters(_validate_multifit_datasets(datasets))
    first = datasets[0].stack_model
    p = build_fit_parameters(first,spline,eps_offset_bounds,eps2_coeff_bounds,eps_offset_parallel_init,eps_offset_perp_init,eps2_coeff_init,previous_result=previous_result,old_spline=old_spline)
    for dataset in datasets[1:]:
        stack_model = dataset.stack_model
        for name,spec in stack_model.thickness_defs.items():
            pname = thickness_parameter_name(stack_model,name)
            value = previous_result.params[pname].value if previous_result is not None and pname in previous_result.params else spec["value"]
            if spec["vary"]:
                p.add(pname,value=np.clip(value,spec["min"],spec["max"]),min=spec["min"],max=spec["max"],vary=True)
            else:
                p.add(pname,value=spec["value"],vary=False)
    return p

def _multi_data_residual(params,optical_model,datasets,weighted=True):
    pieces = []
    for dataset in _validate_multifit_datasets(datasets):
        residual = ellipsometry_residuals(params,optical_model,dataset.stack_model,dataset.measurement)
        pieces.append(_stack_sigma_residual(residual,dataset.measurement,weighted=weighted))
    return np.concatenate(pieces)

def _multi_anchor_residual(params,datasets):
    pieces = []
    for dataset in datasets:
        residual = _thickness_anchor_residual(params,dataset.stack_model,dataset.measurement)
        if residual.size: pieces.append(residual)
    return np.concatenate(pieces) if pieces else np.array([],dtype=float)

def _multi_dielectric_guard_residual(params,optical_model,datasets):
    """Apply the shared dielectric guard once, scaled to the total amount of fitted data."""
    datasets = _validate_multifit_datasets(datasets)
    reference = datasets[0].stack_model
    guard = getattr(reference,"dielectric_guard",{}) or {}
    if not reference.has_unknown or not guard.get("enabled",False) or float(guard.get("weight",0.0)) <= 0:
        return np.array([],dtype=float)
    eps_o,eps_e = dielectric_from_model(params,optical_model,reference,wl_eval_nm=optical_model.wl_nm)
    arrays = [np.sqrt(np.abs(eps_o))]
    if reference.is_uniaxial: arrays.append(np.sqrt(np.abs(eps_e)))
    magnitude = np.concatenate(arrays)
    max_index = float(guard["max_index"])
    transition = float(guard["transition"])
    weight = float(guard["weight"])
    excess = np.maximum(magnitude-max_index,0.0)/transition
    n_data = max(1,sum(2*len(dataset.measurement.AOIs)*len(dataset.measurement.wl_nm) for dataset in datasets))
    return np.sqrt(weight*n_data/max(len(excess),1))*excess

def _multi_regularized_residual(params,optical_model,datasets):
    pieces = [_multi_data_residual(params,optical_model,datasets,weighted=True)]
    anchors = _multi_anchor_residual(params,datasets)
    guard = _multi_dielectric_guard_residual(params,optical_model,datasets)
    if anchors.size: pieces.append(anchors)
    if guard.size: pieces.append(guard)
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

def _annotate_multi_fit(result,optical_model,datasets):
    datasets = _validate_multifit_datasets(datasets)
    weighted = _multi_data_residual(result.params,optical_model,datasets,weighted=True)
    unweighted = _multi_data_residual(result.params,optical_model,datasets,weighted=False)
    data_chisqr = float(np.sum(weighted**2))
    data_ndata = int(weighted.size)
    data_dof = max(data_ndata-int(result.nvarys),1)
    result.data_residual = weighted
    result.sigma_residual = unweighted
    result.data_chisqr = data_chisqr
    result.data_ndata = data_ndata
    result.data_redchi = data_chisqr/data_dof
    result.sigma_rms = float(np.sqrt(np.mean(unweighted**2)))
    result.regularization_chisqr = max(0.0,float(result.chisqr)-data_chisqr)
    result.spectral_weighting = tuple(sorted({dataset.measurement.spectral_weighting for dataset in datasets}))
    metrics = {}
    for dataset in datasets:
        residual = ellipsometry_residuals(result.params,optical_model,dataset.stack_model,dataset.measurement)
        sigma = _stack_sigma_residual(residual,dataset.measurement,weighted=False)
        psi_deg = np.concatenate([residual["psi_deg"][aoi] for aoi in dataset.measurement.AOIs])
        delta_deg = np.concatenate([residual["delta_deg"][aoi] for aoi in dataset.measurement.AOIs])
        metrics[dataset.label] = {
            "rms_sigma":float(np.sqrt(np.mean(sigma**2))),
            "max_sigma":float(np.max(np.abs(sigma))),
            "psi_rms_deg":float(np.sqrt(np.mean(psi_deg**2))),
            "delta_rms_deg":float(np.sqrt(np.mean(delta_deg**2))),
        }
    result.dataset_metrics = metrics
    return result

def fit_spline_multi(params,spline,datasets,max_nfev=3000):
    """Jointly fit one spline dielectric tensor to several flakes."""
    datasets = configure_multifit_thickness_parameters(_validate_multifit_datasets(datasets))
    backend = _prepare_batched_multifit_backend(datasets)
    if backend is None:
        objective = lambda p:_multi_regularized_residual(p,spline,datasets)
    else:
        objective = lambda p:_batched_multifit_regularized_residual(p,spline,backend)
    result = minimize(objective,params,method="least_squares",max_nfev=max_nfev,scale_covar=False)
    result.fast_multifit_backend = backend is not None
    return _annotate_multi_fit(result,spline,datasets)

def multi_dataset_fit_table(result,datasets):
    rows = []
    datasets = list(datasets)
    for dataset in datasets:
        row = {"flake":dataset.label}
        for name in dataset.stack_model.thickness_defs:
            par = result.params[thickness_parameter_name(dataset.stack_model,name)]
            row[f"{name}_nm"] = float(par.value)
            row[f"{name}_vary"] = bool(par.vary)
            row[f"{name}_stderr_nm"] = np.nan if par.stderr is None else float(par.stderr)
        row.update(getattr(result,"dataset_metrics",{}).get(dataset.label,{}))
        rows.append(row)
    return pd.DataFrame(rows)

def build_multi_physical_parameters(datasets,model,previous_result=None,new_term_seed=None,eps_inf_bounds=(-20.0,50.0),eps_inf_parallel_init=3.0,eps_inf_perp_init=2.0):
    """One shared physical dielectric model plus independent thicknesses for every flake."""
    datasets = configure_multifit_thickness_parameters(_validate_multifit_datasets(datasets))
    first = datasets[0].stack_model
    p = build_physical_parameters(
        first,model,previous_result=previous_result,new_term_seed=new_term_seed,
        eps_inf_bounds=eps_inf_bounds,eps_inf_parallel_init=eps_inf_parallel_init,
        eps_inf_perp_init=eps_inf_perp_init,
    )
    for dataset in datasets[1:]:
        stack_model = dataset.stack_model
        for name,spec in stack_model.thickness_defs.items():
            pname = thickness_parameter_name(stack_model,name)
            value = previous_result.params[pname].value if previous_result is not None and pname in previous_result.params else spec["value"]
            if spec["vary"]:
                p.add(pname,value=np.clip(value,spec["min"],spec["max"]),min=spec["min"],max=spec["max"],vary=True)
            else:
                p.add(pname,value=spec["value"],vary=False)
    return p

def fit_physical_multi(params,model,datasets,max_nfev=3000):
    """Jointly fit one physical dielectric tensor to several flakes."""
    if not isinstance(model,PhysicalModel): raise TypeError("model must be a PhysicalModel.")
    datasets = configure_multifit_thickness_parameters(_validate_multifit_datasets(datasets))
    backend = _prepare_batched_multifit_backend(datasets)
    if backend is None:
        objective = lambda p:_multi_regularized_residual(p,model,datasets)
    else:
        objective = lambda p:_batched_multifit_regularized_residual(p,model,backend)
    result = minimize(objective,params,method="least_squares",max_nfev=max_nfev,scale_covar=False)
    result.fast_multifit_backend = backend is not None
    return _annotate_multi_fit(result,model,datasets)

