"""Dielectric-function models, stack construction, and fit-parameter builders."""

import numpy as np
from scipy.interpolate import BSpline
from lmfit import Parameters

from .kramers_kronig import im2re

from .models import PhysicalModel, SplineModel, StackModel, thickness_parameter_name

def _build_spline_basis(wl_nm,internal_knots,degree,kk_e_min,kk_e_max,kk_points,hc_ev_nm):
    """Construct the B-spline basis and its KK transform for a given set of internal knots."""
    wl_nm = np.asarray(wl_nm,dtype=float)
    E_meas = hc_ev_nm/wl_nm
    E_kk = np.linspace(kk_e_min,kk_e_max,kk_points)
    internal_knots = np.sort(np.asarray(internal_knots,dtype=float))
    knot_vector = np.r_[np.full(degree+1,kk_e_min),internal_knots,np.full(degree+1,kk_e_max)]
    n_basis_total = len(knot_vector)-degree-1
    spline = BSpline(knot_vector,np.eye(n_basis_total),degree,extrapolate=False)
    B_kk = spline(E_kk)[:,1:-1]
    B_meas = spline(E_meas)[:,1:-1]
    KK_basis_kk = np.column_stack([im2re(B_kk[:,j],E_kk) for j in range(B_kk.shape[1])])
    KK_basis_meas = np.column_stack([np.interp(E_meas,E_kk,KK_basis_kk[:,j]) for j in range(B_kk.shape[1])])
    return SplineModel(wl_nm,E_meas,E_kk,internal_knots,knot_vector,B_kk,B_meas,KK_basis_kk,KK_basis_meas,degree,kk_e_min,kk_e_max,hc_ev_nm)

def build_initial_spline(wl_nm,bandwidth_nm,degree=3,initial_free_nodes=None,auto_nodes_min=4,auto_nodes_max=7,coarse_resolution_factor=12.0,kk_e_min=None,kk_e_max=None,kk_points=1201,hc_ev_nm=1239.8419843320026):
    """Build coarse initial KK B-spline over measured energy range."""
    wl_nm = np.asarray(wl_nm,dtype=float)
    bandwidth_nm = np.asarray(bandwidth_nm,dtype=float)
    E_meas = hc_ev_nm/wl_nm
    e_min,e_max = float(np.min(E_meas)),float(np.max(E_meas))
    kk_e_min = max(0.05,0.30*e_min) if kk_e_min is None else float(kk_e_min)
    kk_e_max = 2.50*e_max if kk_e_max is None else float(kk_e_max)
    bandwidth_eV = hc_ev_nm*bandwidth_nm/(wl_nm**2)
    if initial_free_nodes is None:
        initial_spacing_eV = coarse_resolution_factor*np.median(bandwidth_eV)
        initial_free_nodes = int(np.ceil((e_max-e_min)/max(initial_spacing_eV,1e-9)))+1
        initial_free_nodes = int(np.clip(initial_free_nodes,auto_nodes_min,auto_nodes_max))
    initial_free_nodes = max(int(initial_free_nodes),degree-1)
    n_initial_internal = max(0,initial_free_nodes-degree+1)
    initial_internal_knots = np.linspace(e_min,e_max,n_initial_internal+2)[1:-1] if n_initial_internal > 0 else np.array([])
    return _build_spline_basis(wl_nm,initial_internal_knots,degree,kk_e_min,kk_e_max,kk_points,hc_ev_nm)

def build_initial_physical_model(wl_nm,kk_e_min=None,kk_e_max=None,kk_points=1201,hc_ev_nm=1239.8419843320026):
    """Create constant-epsilon baseline used before screening physical terms."""
    wl_nm = np.asarray(wl_nm,dtype=float)
    E_meas = hc_ev_nm/wl_nm
    e_min,e_max = float(np.min(E_meas)),float(np.max(E_meas))
    kk_e_min = max(0.01,0.10*e_min) if kk_e_min is None else float(kk_e_min)
    kk_e_max = 3.00*e_max if kk_e_max is None else float(kk_e_max)
    return PhysicalModel(wl_nm,"constant",tuple(),kk_e_min,kk_e_max,int(kk_points),float(hc_ev_nm))

def _canonical_physical_kind(kind):
    kind = str(kind).strip().lower().replace("-","_").replace(" ","_")
    aliases = {"tauc":"tauc_lorentz","tauclorentz":"tauc_lorentz","cody":"cody_lorentz","codylorentz":"cody_lorentz",
        "sellmeier":"sellmeier_uv","uv_sellmeier":"sellmeier_uv","ir_sellmeier":"sellmeier_ir","free_carrier":"drude",}
    return aliases.get(kind,kind)

def add_physical_term(model,kind):
    """Return a physical model with one additional susceptibility or Cauchy background."""
    kind = _canonical_physical_kind(kind)
    if kind == "cauchy":
        if model.background == "cauchy": raise ValueError("A Cauchy background is already present.")
        return PhysicalModel(model.wl_nm,"cauchy",model.terms,model.kk_e_min,model.kk_e_max,model.kk_points,model.hc_ev_nm)
    allowed = ("sellmeier_uv","sellmeier_ir","drude","lorentz","harmonic","gaussian","tauc_lorentz","cody_lorentz")
    if kind not in allowed: raise ValueError(f"Unknown physical term {kind!r}; choose from {('cauchy',)+allowed}.")
    return PhysicalModel(model.wl_nm,model.background,model.terms+(kind,),model.kk_e_min,model.kk_e_max,model.kk_points,model.hc_ev_nm)

def _thickness_definition_nm(value):
    """Convert a scalar or two-element array into a thickness definition dictionary."""
    flat = np.asarray(value,dtype=float).reshape(-1)
    if flat.size == 1:
        value_nm = float(flat[0])*1e9
        return {"vary":False,"value":value_nm,"min":value_nm,"max":value_nm}
    if flat.size != 2: raise ValueError("Layer thickness must be either a scalar or a two-element [lower, upper] array.")
    lo_nm,hi_nm = flat.astype(float)*1e9
    if hi_nm < lo_nm: raise ValueError("Layer thickness upper bound must be >= lower bound.")
    if np.isclose(lo_nm,hi_nm,rtol=0.0,atol=1e-12):
        value_nm = 0.5*(lo_nm+hi_nm)
        return {"vary":False,"value":value_nm,"min":value_nm,"max":value_nm}
    if np.isfinite(lo_nm) and np.isfinite(hi_nm): value_nm = 0.5*(lo_nm+hi_nm)
    elif np.isfinite(lo_nm): value_nm = max(lo_nm+10.0,100.0)
    elif np.isfinite(hi_nm): value_nm = max(1.0,0.5*hi_nm)
    else: value_nm = 100.0
    return {"vary":True,"value":value_nm,"min":float(lo_nm),"max":float(hi_nm)}

def build_stack_model(stack,dielectric_functions,thicknesses,wl_nm,symmetry="uniaxial"):
    """Collect isotropic/uniaxial optical constants, allowing at most one unknown material.
    Each known layer may be supplied as "layer_i" containing an isotropic
    array or an "{"o": ..., "e": ...}" pair.
    """
    symmetry = str(symmetry).lower()
    if symmetry not in ("isotropic","uniaxial"): raise ValueError("symmetry must be either 'isotropic' or 'uniaxial'.")
    names = stack.split("/")
    layer_sources = {}
    for i,name in enumerate(names):
        base_key,o_key,e_key = f"layer_{i}",f"layer_{i}_o",f"layer_{i}_e"
        has_base,has_o,has_e = base_key in dielectric_functions,o_key in dielectric_functions,e_key in dielectric_functions
        if has_base and (has_o or has_e):
            raise ValueError(f"{name}: supply either {base_key} or the {o_key}/{e_key} pair, not both.")
        if has_o != has_e:
            missing = e_key if has_o else o_key
            raise ValueError(f"{name}: incomplete anisotropic dielectric data; missing {missing}.")
        if has_base: layer_sources[i] = dielectric_functions[base_key]
        elif has_o: layer_sources[i] = {"o":dielectric_functions[o_key],"e":dielectric_functions[e_key]}
    known_indices = sorted(layer_sources)
    unknown_indices = [i for i in range(len(names)) if i not in layer_sources]
    if len(unknown_indices) > 1: raise ValueError(f"At most one material may be missing an n,k table, but found {[names[i] for i in unknown_indices]}.")
    unknown_index = unknown_indices[0] if len(unknown_indices) == 1 else None
    if unknown_index is not None and unknown_index in (0,len(names)-1): raise ValueError("A missing optical dataset may only belong to a finite layer, not the incident medium or substrate.")
    unknown_name = names[unknown_index] if unknown_index is not None else None
    wl_nm = np.asarray(wl_nm,dtype=float)
    known_eps,known_eps_e = {},{}
    for i in known_indices:
        source = layer_sources[i]
        if isinstance(source,dict):
            aliases = (("o","e"),("ordinary","extraordinary"),("parallel","perpendicular"))
            pair = next(((left,right) for left,right in aliases if left in source and right in source),None)
            if pair is None: raise ValueError(f"{names[i]}: anisotropic dielectric data must contain a complete o/e pair.")
            eps_o = np.asarray(source[pair[0]],dtype=complex)
            eps_e = np.asarray(source[pair[1]],dtype=complex)
        elif isinstance(source,(tuple,list)) and len(source) == 2:
            eps_o,eps_e = (np.asarray(component,dtype=complex) for component in source)
        else:
            eps_o = eps_e = np.asarray(source,dtype=complex)
        if eps_o.shape != wl_nm.shape: raise ValueError(f"Ordinary/in-plane optical data for {names[i]} has shape {eps_o.shape}, expected {wl_nm.shape}.")
        if eps_e.shape != wl_nm.shape: raise ValueError(f"Extraordinary/out-of-plane optical data for {names[i]} has shape {eps_e.shape}, expected {wl_nm.shape}.")
        known_eps[i],known_eps_e[i] = eps_o,eps_e
    thickness_defs = {name:_thickness_definition_nm(thicknesses[name]) for name in names[1:-1]}
    return StackModel(
        names=names,
        unknown_index=unknown_index,
        unknown_name=unknown_name,
        known_eps=known_eps,
        thickness_defs=thickness_defs,
        wl_nm=wl_nm,
        symmetry=symmetry,
        known_eps_e=known_eps_e,
    )

def build_thickness_parameters(stack_model):
    """Create only the finite-layer thickness parameters for a fully known optical stack."""
    if stack_model.has_unknown: raise ValueError("build_thickness_parameters() is only used when all material optical constants are known.")
    p = Parameters()
    for name,spec in stack_model.thickness_defs.items():
        pname = thickness_parameter_name(stack_model,name)
        if spec["vary"]: p.add(pname,value=spec["value"],min=spec["min"],max=spec["max"],vary=True)
        else: p.add(pname,value=spec["value"],vary=False)
    return p

def build_fit_parameters(stack_model,spline,eps_offset_bounds=(-np.inf,np.inf),eps2_coeff_bounds=(0.0,np.inf),eps_offset_parallel_init=4.0,eps_offset_perp_init=2.5,eps2_coeff_init=1e-3,previous_result=None,old_spline=None):
    """Create spline and thickness fit parameters, using one dielectric function for isotropic media and two for uniaxial media."""
    if not stack_model.has_unknown: raise ValueError("build_fit_parameters() is only used when one material optical dataset is missing.")
    p = Parameters()
    unknown_name = stack_model.unknown_name

    if stack_model.is_isotropic:
        if previous_result is None:
            offset = eps_offset_parallel_init
            coeff = np.full(spline.n_free_nodes,eps2_coeff_init)
        else:
            if old_spline is None: raise ValueError("old_spline is required when inheriting parameters from a previous fit.")
            old_p = previous_result.params
            old_coeff = np.array([old_p[f"{unknown_name}_eps2_{j}"].value for j in range(old_spline.n_free_nodes)])
            coeff = np.clip(np.linalg.lstsq(spline.B_kk,old_spline.B_kk@old_coeff,rcond=None)[0],eps2_coeff_bounds[0],eps2_coeff_bounds[1])
            offset = old_p[f"{unknown_name}_eps_offset"].value
        offset_name = f"{unknown_name}_eps_offset"
        offset_bounds = (old_p[offset_name].min,old_p[offset_name].max) if previous_result is not None and offset_name in old_p else eps_offset_bounds
        coeff_bounds = (old_p[f"{unknown_name}_eps2_0"].min,old_p[f"{unknown_name}_eps2_0"].max) if previous_result is not None and f"{unknown_name}_eps2_0" in old_p else eps2_coeff_bounds
        p.add(offset_name,value=offset,min=offset_bounds[0],max=offset_bounds[1])
        for j in range(spline.n_free_nodes): p.add(f"{unknown_name}_eps2_{j}",value=coeff[j],min=coeff_bounds[0],max=coeff_bounds[1])

    else:
        if previous_result is None:
            offset_o,offset_e = eps_offset_parallel_init,eps_offset_perp_init
            coeff_o = np.full(spline.n_free_nodes,eps2_coeff_init)
            coeff_e = np.full(spline.n_free_nodes,eps2_coeff_init)
        else:
            if old_spline is None: raise ValueError("old_spline is required when inheriting parameters from a previous fit.")
            old_p = previous_result.params
            old_coeff_o = np.array([old_p[f"{unknown_name}_eps2_o_{j}"].value for j in range(old_spline.n_free_nodes)])
            old_coeff_e = np.array([old_p[f"{unknown_name}_eps2_e_{j}"].value for j in range(old_spline.n_free_nodes)])
            coeff_o = np.clip(np.linalg.lstsq(spline.B_kk,old_spline.B_kk@old_coeff_o,rcond=None)[0],eps2_coeff_bounds[0],eps2_coeff_bounds[1])
            coeff_e = np.clip(np.linalg.lstsq(spline.B_kk,old_spline.B_kk@old_coeff_e,rcond=None)[0],eps2_coeff_bounds[0],eps2_coeff_bounds[1])
            offset_o = old_p[f"{unknown_name}_eps_offset_o"].value
            offset_e = old_p[f"{unknown_name}_eps_offset_e"].value
        offset_o_name,offset_e_name = f"{unknown_name}_eps_offset_o",f"{unknown_name}_eps_offset_e"
        if previous_result is not None:
            offset_o_bounds = (old_p[offset_o_name].min,old_p[offset_o_name].max)
            offset_e_bounds = (old_p[offset_e_name].min,old_p[offset_e_name].max)
            coeff_o_bounds = (old_p[f"{unknown_name}_eps2_o_0"].min,old_p[f"{unknown_name}_eps2_o_0"].max)
            coeff_e_bounds = (old_p[f"{unknown_name}_eps2_e_0"].min,old_p[f"{unknown_name}_eps2_e_0"].max)
        else:
            offset_o_bounds = offset_e_bounds = eps_offset_bounds
            coeff_o_bounds = coeff_e_bounds = eps2_coeff_bounds
        p.add(offset_o_name,value=offset_o,min=offset_o_bounds[0],max=offset_o_bounds[1])
        p.add(offset_e_name,value=offset_e,min=offset_e_bounds[0],max=offset_e_bounds[1])
        for j in range(spline.n_free_nodes):
            p.add(f"{unknown_name}_eps2_o_{j}",value=coeff_o[j],min=coeff_o_bounds[0],max=coeff_o_bounds[1])
            p.add(f"{unknown_name}_eps2_e_{j}",value=coeff_e[j],min=coeff_e_bounds[0],max=coeff_e_bounds[1])

    for name,spec in stack_model.thickness_defs.items():
        pname = thickness_parameter_name(stack_model,name)
        value = previous_result.params[pname].value if previous_result is not None and pname in previous_result.params else spec["value"]
        if spec["vary"]: p.add(pname,value=np.clip(value,spec["min"],spec["max"]),min=spec["min"],max=spec["max"],vary=True)
        else: p.add(pname,value=spec["value"],vary=False)
    return p

def _physical_direction_prefix(stack_model,direction):
    base = f"{stack_model.unknown_name}_phys"
    return base if stack_model.is_isotropic else f"{base}_{direction}"

def build_physical_parameters(stack_model,model,previous_result=None,new_term_seed=None,eps_inf_bounds=(-20.0,50.0),eps_inf_parallel_init=3.0,eps_inf_perp_init=2.0):
    """Build bounded physical-model parameters while inheriting accepted terms."""
    if not stack_model.has_unknown: raise ValueError("Physical parameters require one unknown finite layer.")
    if not isinstance(model,PhysicalModel): raise TypeError("model must be a PhysicalModel.")
    seed = {} if new_term_seed is None else dict(new_term_seed)
    p = Parameters()
    E_meas = model.hc_ev_nm/np.asarray(model.wl_nm,dtype=float)
    e_min,e_max = float(np.min(E_meas)),float(np.max(E_meas))
    e_span = max(e_max-e_min,0.1)
    wl_min,wl_max = float(np.min(model.wl_nm)),float(np.max(model.wl_nm))
    directions = ("iso",) if stack_model.is_isotropic else ("o","e")

    def inherited(name,default):
        if previous_result is not None and name in previous_result.params:
            return float(previous_result.params[name].value)
        return float(default)

    for direction in directions:
        prefix = _physical_direction_prefix(stack_model,direction)
        eps_default = eps_inf_parallel_init if direction in ("iso","o") else eps_inf_perp_init
        if model.background == "constant":
            name = f"{prefix}_eps_inf"
            p.add(name,value=np.clip(inherited(name,eps_default),*eps_inf_bounds),min=eps_inf_bounds[0],max=eps_inf_bounds[1])
        elif model.background == "cauchy":
            eps_name = f"{prefix}_eps_inf"
            n_default = np.sqrt(max(inherited(eps_name,eps_default),0.01))
            background_specs = {
                "A":(n_default,0.1,10.0),
                "B":(0.01,0.0,2.0),
                "C":(0.001,0.0,1.0),
            }
            for par,(default,lo,hi) in background_specs.items():
                name = f"{prefix}_cauchy_{par}"
                p.add(name,value=np.clip(inherited(name,seed.get(par,default)),lo,hi),min=lo,max=hi)
        else:
            raise ValueError(f"Unsupported physical background {model.background!r}.")

        for j,kind in enumerate(model.terms):
            term = f"{prefix}_{j}_{kind}"
            probe = "strength" if kind.startswith("sellmeier") else "amp"
            is_new = j == len(model.terms)-1 and (previous_result is None or f"{term}_{probe}" not in previous_result.params)
            local_seed = seed if is_new else {}
            if kind in ("sellmeier_uv","sellmeier_ir"):
                if kind == "sellmeier_uv":
                    pole_default,pole_bounds = 0.55*wl_min,(1.0,max(1.01,0.98*wl_min))
                else:
                    pole_default,pole_bounds = 2.0*wl_max,(1.02*wl_max,100.0*wl_max)
                defaults = {"strength":1.0,"lambda0_nm":pole_default}
                bounds = {"strength":(0.0,500.0),"lambda0_nm":pole_bounds}
            elif kind == "drude":
                defaults = {"amp":max(0.1,e_min**2),"gamma":max(0.01,0.1*e_min)}
                bounds = {"amp":(0.0,1e4),"gamma":(1e-4,5.0*e_max)}
            elif kind in ("lorentz","harmonic","gaussian"):
                defaults = {"amp":5.0,"e0":0.5*(e_min+e_max),"gamma":max(0.03,0.15*e_span)}
                bounds = {"amp":(0.0,1e4),"e0":(max(0.01,0.25*e_min),3.0*e_max),"gamma":(1e-3,5.0*e_max)}
            elif kind == "tauc_lorentz":
                defaults = {"amp":20.0,"eg":max(0.0,0.65*e_min),"de":max(0.05,0.35*e_span),"gamma":max(0.03,0.20*e_span)}
                bounds = {"amp":(0.0,1e4),"eg":(0.0,1.2*e_max),"de":(0.01,3.0*e_max),"gamma":(1e-3,5.0*e_max)}
            elif kind == "cody_lorentz":
                defaults = {"amp":20.0,"eg":max(0.0,0.65*e_min),"de":max(0.05,0.35*e_span),"gamma":max(0.03,0.20*e_span),"ep":max(0.03,0.15*e_span)}
                bounds = {"amp":(0.0,1e4),"eg":(0.0,1.2*e_max),"de":(0.01,3.0*e_max),"gamma":(1e-3,5.0*e_max),"ep":(1e-3,5.0*e_max)}
            else:
                raise ValueError(f"Unsupported physical term {kind!r}.")
            for parameter,default in defaults.items():
                name = f"{term}_{parameter}"
                value = inherited(name,local_seed.get(parameter,default))
                lo,hi = bounds[parameter]
                p.add(name,value=np.clip(value,lo,hi),min=lo,max=hi)

    for name,spec in stack_model.thickness_defs.items():
        pname = thickness_parameter_name(stack_model,name)
        value = inherited(pname,spec["value"])
        if spec["vary"]: p.add(pname,value=np.clip(value,spec["min"],spec["max"]),min=spec["min"],max=spec["max"],vary=True)
        else: p.add(pname,value=spec["value"],vary=False)
    return p

def dielectric_from_spline(params,spline,stack_model,wl_eval_nm=None):
    """Evaluate the reconstructed dielectric function. For isotropic media the returned parallel/perpendicular arrays are identical."""
    if not stack_model.has_unknown: raise ValueError("No dielectric function is being reconstructed because every material is known.")
    unknown_name = stack_model.unknown_name

    if wl_eval_nm is None:
        B_eval,KK_eval = spline.B_meas,spline.KK_basis_meas
    else:
        wl_eval_nm = np.asarray(wl_eval_nm,dtype=float)
        E_eval = spline.hc_ev_nm/wl_eval_nm
        if np.any((E_eval < spline.kk_e_min) | (E_eval > spline.kk_e_max)): raise ValueError("Evaluation wavelengths extend outside the KK spline domain.")
        n_basis_total = len(spline.knot_vector)-spline.degree-1
        bspline = BSpline(spline.knot_vector,np.eye(n_basis_total),spline.degree,extrapolate=False)
        B_eval = bspline(E_eval)[:,1:-1]
        KK_eval = np.column_stack([np.interp(E_eval,spline.E_kk,spline.KK_basis_kk[:,j]) for j in range(spline.n_free_nodes)])

    if stack_model.is_isotropic:
        coeff = np.array([params[f"{unknown_name}_eps2_{j}"].value for j in range(spline.n_free_nodes)])
        eps = params[f"{unknown_name}_eps_offset"].value+KK_eval@coeff+1j*(B_eval@coeff)
        return eps,eps

    coeff_o = np.array([params[f"{unknown_name}_eps2_o_{j}"].value for j in range(spline.n_free_nodes)])
    coeff_e = np.array([params[f"{unknown_name}_eps2_e_{j}"].value for j in range(spline.n_free_nodes)])
    eps_o = params[f"{unknown_name}_eps_offset_o"].value+KK_eval@coeff_o+1j*(B_eval@coeff_o)
    eps_e = params[f"{unknown_name}_eps_offset_e"].value+KK_eval@coeff_e+1j*(B_eval@coeff_e)
    return eps_o,eps_e

def _physical_direction_epsilon(params,model,stack_model,direction,wl_eval_nm):
    wl_eval_nm = np.asarray(wl_eval_nm,dtype=float)
    E_eval = model.hc_ev_nm/wl_eval_nm
    if np.any((E_eval < model.kk_e_min) | (E_eval > model.kk_e_max)):
        raise ValueError("Evaluation wavelengths extend outside the physical model's KK domain.")
    prefix = _physical_direction_prefix(stack_model,direction)
    if model.background == "constant":
        eps = np.full(wl_eval_nm.shape,params[f"{prefix}_eps_inf"].value,dtype=complex)
    elif model.background == "cauchy":
        wl_um = wl_eval_nm/1000.0
        A = params[f"{prefix}_cauchy_A"].value
        B = params[f"{prefix}_cauchy_B"].value
        C = params[f"{prefix}_cauchy_C"].value
        eps = (A+B/wl_um**2+C/wl_um**4).astype(complex)**2
    else:
        raise ValueError(f"Unsupported physical background {model.background!r}.")

    E_kk = np.linspace(model.kk_e_min,model.kk_e_max,model.kk_points)
    eps2_kk = np.zeros_like(E_kk)
    has_kk_term = False
    for j,kind in enumerate(model.terms):
        term = f"{prefix}_{j}_{kind}"
        if kind in ("sellmeier_uv","sellmeier_ir"):
            strength = params[f"{term}_strength"].value
            lambda0_nm = params[f"{term}_lambda0_nm"].value
            eps += strength*wl_eval_nm**2/(wl_eval_nm**2-lambda0_nm**2)
        elif kind == "drude":
            amp = params[f"{term}_amp"].value
            gamma = params[f"{term}_gamma"].value
            eps += -amp/(E_eval**2+1j*gamma*E_eval)
        elif kind in ("lorentz","harmonic"):
            amp = params[f"{term}_amp"].value
            e0 = params[f"{term}_e0"].value
            gamma = params[f"{term}_gamma"].value
            harmonic_shift = 0.25*gamma**2 if kind == "harmonic" else 0.0
            eps += amp*gamma*e0/(e0**2-E_eval**2+harmonic_shift-1j*gamma*E_eval)
        elif kind == "gaussian":
            amp = params[f"{term}_amp"].value
            e0 = params[f"{term}_e0"].value
            gamma = params[f"{term}_gamma"].value
            sigma = gamma/(2.0*np.sqrt(np.log(2.0)))
            eps2_kk += amp*(np.exp(-((E_kk-e0)/sigma)**2)-np.exp(-((E_kk+e0)/sigma)**2))
            has_kk_term = True
        elif kind == "tauc_lorentz":
            amp = params[f"{term}_amp"].value
            eg = params[f"{term}_eg"].value
            e0 = eg+params[f"{term}_de"].value
            gamma = params[f"{term}_gamma"].value
            active = E_kk > eg
            Ea = E_kk[active]
            denominator = Ea*((Ea**2-e0**2)**2+(gamma*Ea)**2)
            eps2_kk[active] += amp*gamma*e0*(Ea-eg)**2/np.maximum(denominator,1e-30)
            has_kk_term = True
        elif kind == "cody_lorentz":
            amp = params[f"{term}_amp"].value
            eg = params[f"{term}_eg"].value
            e0 = eg+params[f"{term}_de"].value
            gamma = params[f"{term}_gamma"].value
            ep = params[f"{term}_ep"].value
            active = E_kk > eg
            Ea = E_kk[active]
            cody = (Ea-eg)**2/((Ea-eg)**2+ep**2)
            lorentz = amp*e0*gamma*Ea/((Ea**2-e0**2)**2+(gamma*Ea)**2)
            eps2_kk[active] += cody*lorentz
            has_kk_term = True
        else:
            raise ValueError(f"Unsupported physical term {kind!r}.")
    if has_kk_term:
        eps1_kk = im2re(eps2_kk,E_kk)
        eps += np.interp(E_eval,E_kk,eps1_kk)+1j*np.interp(E_eval,E_kk,eps2_kk)
    return eps

def dielectric_from_physical(params,model,stack_model,wl_eval_nm=None):
    """Evaluate a Cauchy/constant background and all selected physical terms."""
    if not stack_model.has_unknown: raise ValueError("No dielectric function is being reconstructed because every material is known.")
    wl_eval_nm = model.wl_nm if wl_eval_nm is None else np.asarray(wl_eval_nm,dtype=float)
    if stack_model.is_isotropic:
        eps = _physical_direction_epsilon(params,model,stack_model,"iso",wl_eval_nm)
        return eps,eps
    eps_o = _physical_direction_epsilon(params,model,stack_model,"o",wl_eval_nm)
    eps_e = _physical_direction_epsilon(params,model,stack_model,"e",wl_eval_nm)
    return eps_o,eps_e

def dielectric_from_model(params,optical_model,stack_model,wl_eval_nm=None):
    """Dispatch dielectric evaluation without changing the Fresnel API."""
    if isinstance(optical_model,SplineModel):
        return dielectric_from_spline(params,optical_model,stack_model,wl_eval_nm=wl_eval_nm)
    if isinstance(optical_model,PhysicalModel):
        return dielectric_from_physical(params,optical_model,stack_model,wl_eval_nm=wl_eval_nm)
    raise TypeError("optical_model must be a SplineModel or PhysicalModel.")

