"""Symmetry-reduced Fresnel forward model for isotropic/uniaxial planar stacks."""

import numpy as np
try:
    from numba import njit
except ImportError:  # pragma: no cover - pure NumPy fallback remains available
    njit = None

from .dielectric import dielectric_from_model
from .models import thickness_parameter_name

def _forward_sqrt(z):
    q = np.sqrt(np.asarray(z,dtype=complex))
    flip = (q.imag < 0) | ((np.abs(q.imag) < 1e-14) & (q.real < 0))
    return np.where(flip,-q,q)

def _get_forward_cache(stack_model,aois):
    """Cache every AOI/wavelength quantity belonging only to known layers."""
    aois = tuple(float(aoi) for aoi in aois)
    cached = stack_model._forward_cache.get(aois)
    if cached is not None:
        return cached

    n_aoi = len(aois)
    n_layers = len(stack_model.names)
    n_wl = len(stack_model.wl_nm)
    theta = np.deg2rad(np.asarray(aois,dtype=float))[:,None]
    incident_root = _forward_sqrt(stack_model.known_eps[0])[None,:]
    q_parallel = incident_root*np.sin(theta)
    q2 = q_parallel*q_parallel
    qz_s = np.empty((n_aoi,n_layers,n_wl),dtype=complex)
    qz_p = np.empty_like(qz_s)
    y_s = np.empty_like(qz_s)
    y_p = np.empty_like(qz_s)

    for i,eps_o in stack_model.known_eps.items():
        eps_o = np.asarray(eps_o,dtype=complex)[None,:]
        eps_e = np.asarray(stack_model.known_eps_e.get(i,eps_o),dtype=complex)
        if eps_e.ndim == 1: eps_e = eps_e[None,:]
        qz_s[:,i,:] = _forward_sqrt(eps_o-q2)
        qz_p[:,i,:] = _forward_sqrt(eps_o*(1.0-q2/eps_e))
        y_s[:,i,:] = qz_s[:,i,:]
        y_p[:,i,:] = qz_p[:,i,:]/eps_o

    cached = {
        "aois":aois,
        "q2":q2,
        "qz_s":qz_s,
        "qz_p":qz_p,
        "y_s":y_s,
        "y_p":y_p,
        "k0":2*np.pi/np.asarray(stack_model.wl_nm,dtype=float)[None,:],
    }
    stack_model._forward_cache[aois] = cached
    return cached

if njit is not None:
    @njit(cache=True)
    def _forward_sqrt_numba(z):
        q = np.sqrt(z)
        if q.imag < 0.0 or (abs(q.imag) < 1e-14 and q.real < 0.0): q = -q
        return q

    @njit(cache=True)
    def _fresnel_core_numba(q2,qz_s_known,qz_p_known,y_s_known,y_p_known,k0,eps_o,eps_e,d_nm,unknown_index):
        n_aoi,n_layers,n_wl = qz_s_known.shape
        r_s_out = np.empty((n_aoi,n_wl),dtype=np.complex128)
        r_p_out = np.empty((n_aoi,n_wl),dtype=np.complex128)
        rho_out = np.empty((n_aoi,n_wl),dtype=np.complex128)
        psi_out = np.empty((n_aoi,n_wl),dtype=np.float64)
        delta_out = np.empty((n_aoi,n_wl),dtype=np.float64)
        has_unknown = unknown_index >= 0
        for a in range(n_aoi):
            for w in range(n_wl):
                qz_s_unknown = 0j
                qz_p_unknown = 0j
                y_s_unknown = 0j
                y_p_unknown = 0j
                if has_unknown:
                    eps_o_w = eps_o[w]
                    eps_e_w = eps_e[w]
                    qz_s_unknown = _forward_sqrt_numba(eps_o_w-q2[a,w])
                    qz_p_unknown = _forward_sqrt_numba(eps_o_w*(1.0-q2[a,w]/eps_e_w))
                    y_s_unknown = qz_s_unknown
                    y_p_unknown = qz_p_unknown/eps_o_w

                j = n_layers-2
                y_s_left = y_s_unknown if j == unknown_index else y_s_known[a,j,w]
                y_s_right = y_s_unknown if j+1 == unknown_index else y_s_known[a,j+1,w]
                y_p_left = y_p_unknown if j == unknown_index else y_p_known[a,j,w]
                y_p_right = y_p_unknown if j+1 == unknown_index else y_p_known[a,j+1,w]
                r_s = (y_s_left-y_s_right)/(y_s_left+y_s_right)
                r_p = (y_p_left-y_p_right)/(y_p_left+y_p_right)

                for j in range(n_layers-3,-1,-1):
                    layer_index = j+1
                    y_s_left = y_s_unknown if j == unknown_index else y_s_known[a,j,w]
                    y_s_right = y_s_unknown if layer_index == unknown_index else y_s_known[a,layer_index,w]
                    y_p_left = y_p_unknown if j == unknown_index else y_p_known[a,j,w]
                    y_p_right = y_p_unknown if layer_index == unknown_index else y_p_known[a,layer_index,w]
                    r_s_interface = (y_s_left-y_s_right)/(y_s_left+y_s_right)
                    r_p_interface = (y_p_left-y_p_right)/(y_p_left+y_p_right)
                    qz_s_layer = qz_s_unknown if layer_index == unknown_index else qz_s_known[a,layer_index,w]
                    qz_p_layer = qz_p_unknown if layer_index == unknown_index else qz_p_known[a,layer_index,w]
                    phase_s = np.exp(2j*k0[w]*qz_s_layer*d_nm[layer_index])
                    phase_p = np.exp(2j*k0[w]*qz_p_layer*d_nm[layer_index])
                    r_s = (r_s_interface+r_s*phase_s)/(1.0+r_s_interface*r_s*phase_s)
                    r_p = (r_p_interface+r_p*phase_p)/(1.0+r_p_interface*r_p*phase_p)

                rho = r_p/r_s
                r_s_out[a,w] = r_s
                r_p_out[a,w] = r_p
                rho_out[a,w] = rho
                psi_out[a,w] = np.arctan(abs(rho))*180.0/np.pi
                delta_out[a,w] = (-np.arctan2(rho.imag,rho.real)*180.0/np.pi)%360.0
        return r_s_out,r_p_out,rho_out,psi_out,delta_out
    @njit(cache=True)
    def _multiflake_residual_core_numba(q2,qz_s_known,qz_p_known,y_s_known,y_p_known,k0,eps_o,eps_e,d_all_nm,unknown_index,
                                        psi_exp,delta_exp,psi_sigma,delta_sigma,sqrt_weights):
        n_flakes = d_all_nm.shape[0]
        n_aoi,n_layers,n_wl = qz_s_known.shape
        residual = np.empty(n_flakes*n_aoi*2*n_wl,dtype=np.float64)
        has_unknown = unknown_index >= 0
        for f in range(n_flakes):
            flake_offset = f*n_aoi*2*n_wl
            for a in range(n_aoi):
                aoi_offset = flake_offset+a*2*n_wl
                for w in range(n_wl):
                    qz_s_unknown = 0j
                    qz_p_unknown = 0j
                    y_s_unknown = 0j
                    y_p_unknown = 0j
                    if has_unknown:
                        eps_o_w = eps_o[w]
                        eps_e_w = eps_e[w]
                        qz_s_unknown = _forward_sqrt_numba(eps_o_w-q2[a,w])
                        qz_p_unknown = _forward_sqrt_numba(eps_o_w*(1.0-q2[a,w]/eps_e_w))
                        y_s_unknown = qz_s_unknown
                        y_p_unknown = qz_p_unknown/eps_o_w

                    j = n_layers-2
                    y_s_left = y_s_unknown if j == unknown_index else y_s_known[a,j,w]
                    y_s_right = y_s_unknown if j+1 == unknown_index else y_s_known[a,j+1,w]
                    y_p_left = y_p_unknown if j == unknown_index else y_p_known[a,j,w]
                    y_p_right = y_p_unknown if j+1 == unknown_index else y_p_known[a,j+1,w]
                    r_s = (y_s_left-y_s_right)/(y_s_left+y_s_right)
                    r_p = (y_p_left-y_p_right)/(y_p_left+y_p_right)

                    for j in range(n_layers-3,-1,-1):
                        layer_index = j+1
                        y_s_left = y_s_unknown if j == unknown_index else y_s_known[a,j,w]
                        y_s_right = y_s_unknown if layer_index == unknown_index else y_s_known[a,layer_index,w]
                        y_p_left = y_p_unknown if j == unknown_index else y_p_known[a,j,w]
                        y_p_right = y_p_unknown if layer_index == unknown_index else y_p_known[a,layer_index,w]
                        r_s_interface = (y_s_left-y_s_right)/(y_s_left+y_s_right)
                        r_p_interface = (y_p_left-y_p_right)/(y_p_left+y_p_right)
                        qz_s_layer = qz_s_unknown if layer_index == unknown_index else qz_s_known[a,layer_index,w]
                        qz_p_layer = qz_p_unknown if layer_index == unknown_index else qz_p_known[a,layer_index,w]
                        phase_s = np.exp(2j*k0[w]*qz_s_layer*d_all_nm[f,layer_index])
                        phase_p = np.exp(2j*k0[w]*qz_p_layer*d_all_nm[f,layer_index])
                        r_s = (r_s_interface+r_s*phase_s)/(1.0+r_s_interface*r_s*phase_s)
                        r_p = (r_p_interface+r_p*phase_p)/(1.0+r_p_interface*r_p*phase_p)

                    rho = r_p/r_s
                    psi = np.arctan(abs(rho))*180.0/np.pi
                    delta = (-np.arctan2(rho.imag,rho.real)*180.0/np.pi)%360.0
                    dpsi = (psi-psi_exp[f,a,w])/psi_sigma[f,a,w]
                    ddelta = (delta-delta_exp[f,a,w]+180.0)%360.0-180.0
                    ddelta = ddelta/delta_sigma[f,a,w]
                    weight = sqrt_weights[f,w]
                    residual[aoi_offset+w] = dpsi*weight
                    residual[aoi_offset+n_wl+w] = ddelta*weight
        return residual
else:
    _fresnel_core_numba = None
    _multiflake_residual_core_numba = None

def fresnel_reflection_all(params,aois,optical_model,stack_model):
    """Evaluate every AOI together, using the JIT Fresnel core when Numba is available."""
    aois = tuple(float(aoi) for aoi in aois)
    if len(aois) == 0:
        return {}
    cache = _get_forward_cache(stack_model,aois)

    if _fresnel_core_numba is not None:
        if stack_model.has_unknown:
            if optical_model is None: raise ValueError("An optical model is required when the stack contains an unknown material.")
            eps_o,eps_e = dielectric_from_model(params,optical_model,stack_model,wl_eval_nm=stack_model.wl_nm)
            eps_o = np.asarray(eps_o,dtype=np.complex128)
            eps_e = np.asarray(eps_e,dtype=np.complex128)
            unknown_index = int(stack_model.unknown_index)
        else:
            eps_o = np.ones(len(stack_model.wl_nm),dtype=np.complex128)
            eps_e = eps_o
            unknown_index = -1
        d_nm = np.zeros(len(stack_model.names),dtype=float)
        for i,name in enumerate(stack_model.names[1:-1],1): d_nm[i] = params[thickness_parameter_name(stack_model,name)].value
        r_s,r_p,rho,psi,delta = _fresnel_core_numba(
            cache["q2"],cache["qz_s"],cache["qz_p"],cache["y_s"],cache["y_p"],cache["k0"][0],
            eps_o,eps_e,d_nm,unknown_index,
        )
        return {
            aoi:{"r_s":r_s[i],"r_p":r_p[i],"rho":rho[i],"psi":psi[i],"delta":delta[i]}
            for i,aoi in enumerate(aois)
        }

    if stack_model.has_unknown:
        if optical_model is None: raise ValueError("An optical model is required when the stack contains an unknown material.")
        eps_o,eps_e = dielectric_from_model(params,optical_model,stack_model,wl_eval_nm=stack_model.wl_nm)
        qz_s = cache["qz_s"].copy()
        qz_p = cache["qz_p"].copy()
        y_s = cache["y_s"].copy()
        y_p = cache["y_p"].copy()
        i = stack_model.unknown_index
        eps_o = np.asarray(eps_o,dtype=complex)[None,:]
        eps_e = np.asarray(eps_e,dtype=complex)[None,:]
        qz_s[:,i,:] = _forward_sqrt(eps_o-cache["q2"])
        qz_p[:,i,:] = _forward_sqrt(eps_o*(1.0-cache["q2"]/eps_e))
        y_s[:,i,:] = qz_s[:,i,:]
        y_p[:,i,:] = qz_p[:,i,:]/eps_o
    else:
        qz_s,qz_p = cache["qz_s"],cache["qz_p"]
        y_s,y_p = cache["y_s"],cache["y_p"]

    r_s_interfaces = (y_s[:,:-1,:]-y_s[:,1:,:])/(y_s[:,:-1,:]+y_s[:,1:,:])
    r_p_interfaces = (y_p[:,:-1,:]-y_p[:,1:,:])/(y_p[:,:-1,:]+y_p[:,1:,:])
    r_s,r_p = r_s_interfaces[:,-1,:],r_p_interfaces[:,-1,:]
    for j in range(len(stack_model.names)-3,-1,-1):
        layer_index = j+1
        d_nm = params[thickness_parameter_name(stack_model,stack_model.names[layer_index])].value
        phase_s = np.exp(2j*cache["k0"]*qz_s[:,layer_index,:]*d_nm)
        phase_p = np.exp(2j*cache["k0"]*qz_p[:,layer_index,:]*d_nm)
        r_s = (r_s_interfaces[:,j,:]+r_s*phase_s)/(1+r_s_interfaces[:,j,:]*r_s*phase_s)
        r_p = (r_p_interfaces[:,j,:]+r_p*phase_p)/(1+r_p_interfaces[:,j,:]*r_p*phase_p)

    rho = r_p/r_s
    psi = np.rad2deg(np.arctan(np.abs(rho)))
    delta = np.mod(-np.rad2deg(np.angle(rho)),360.0)
    return {
        aoi:{"r_s":r_s[i],"r_p":r_p[i],"rho":rho[i],"psi":psi[i],"delta":delta[i]}
        for i,aoi in enumerate(aois)
    }

def wrap_delta_deg(delta):
    return (delta+180.0)%360.0-180.0

