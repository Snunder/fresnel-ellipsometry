"""Reporting helpers and conversion of fitted dielectric functions to refractive indices."""

import numpy as np
import pandas as pd

from .models import PhysicalModel, thickness_parameter_name
from .dielectric import _physical_direction_prefix, dielectric_from_model



def _physical_model_display_name(model):
    names = {
        "constant": "constant",
        "cauchy": "Cauchy",
        "sellmeier_uv": "Sellmeier (UV)",
        "sellmeier_ir": "Sellmeier (IR)",
        "drude": "Drude",
        "lorentz": "Lorentz",
        "harmonic": "harmonic oscillator",
        "gaussian": "Gaussian",
        "tauc_lorentz": "Tauc--Lorentz",
        "cody_lorentz": "Cody--Lorentz",
    }
    pieces = [names[model.background]] + [names[kind] for kind in model.terms]
    return " + ".join(pieces)


def physical_fit_latex(fit_result, model, stack_model, precision=6):
    r"""Return a report-ready wavelength-domain description of a physical fit.

    Reporting follows two deliberately simple rules:

    * If the fitted model is transparent, print one explicit expression for
      :math:`n(\lambda)`.
    * If any absorptive term is present, print the explicit
      :math:`\varepsilon''(\lambda)` lineshape first, followed by
      :math:`\varepsilon'(\lambda)` written as the non-absorptive real
      contribution plus the Kramers--Kronig transform of
      :math:`\varepsilon''(\lambda)`. Refractive-index expressions are not
      displayed for absorptive models.

    The wavelength variable is always written simply as :math:`\lambda`; no
    variable name encodes a special input unit such as ``lambda_nm``. For direct
    numerical evaluation, :math:`\lambda` denotes the physical wavelength rather
    than a unit-specific number. Fitted coefficients are reported in physically
    convenient units (for example eV for energies, nm for resonance wavelengths,
    and :math:`\mu\mathrm{m}^2` for Cauchy coefficients), with the unit written
    explicitly after every dimensional coefficient. In expressions containing
    :math:`hc/\lambda`, :math:`hc` is given in eV m so that using :math:`\lambda`
    in metres yields the photon energy in eV. Gaussian, Tauc--Lorentz and Cody--Lorentz
    terms are the models for which the implementation constructs the corresponding
    dispersive contribution to :math:`\varepsilon'` numerically through the
    Kramers--Kronig relation.
    """
    if not isinstance(model, PhysicalModel):
        raise TypeError("model must be a PhysicalModel.")

    params = fit_result.params
    fmt = lambda value: f"{float(value):.{int(precision)}g}"
    reports = {}
    directions = (("iso", ""),) if stack_model.is_isotropic else (("o", r"\parallel"), ("e", r"\perp"))
    absorptive_kinds = {"drude", "lorentz", "harmonic", "gaussian", "tauc_lorentz", "cody_lorentz"}

    for direction, subscript in directions:
        prefix = _physical_direction_prefix(stack_model, direction)
        coefficients = []
        transparent_terms = []
        absorption_terms = []
        present_numerical_kk = []

        def add_coefficient(symbol, value, unit=""):
            coefficients.append((symbol, float(value), unit))
            return symbol

        def term_subscript(j):
            return str(j)

        # Real non-absorptive contribution. Its coefficients are still reported for
        # absorptive models, but it is not mixed into the displayed epsilon'' formula.
        cauchy_n = None
        if model.background == "constant":
            eps_inf = add_coefficient(r"\varepsilon_\infty", params[f"{prefix}_eps_inf"].value)
            background_eps = eps_inf
        elif model.background == "cauchy":
            A = add_coefficient("A", params[f"{prefix}_cauchy_A"].value)
            B = add_coefficient("B", params[f"{prefix}_cauchy_B"].value, r"\mu\mathrm{m}^2")
            C = add_coefficient("C", params[f"{prefix}_cauchy_C"].value, r"\mu\mathrm{m}^4")
            cauchy_n = rf"{A}+\frac{{{B}}}{{\lambda^2}}+\frac{{{C}}}{{\lambda^4}}"
            background_eps = rf"\left({cauchy_n}\right)^2"
        else:
            raise ValueError(f"Unsupported physical background {model.background!r}.")

        transparent_terms.append(background_eps)

        for j, kind in enumerate(model.terms, 1):
            term = f"{prefix}_{j-1}_{kind}"
            s = term_subscript(j)

            if kind in ("sellmeier_uv", "sellmeier_ir"):
                S = add_coefficient(rf"S_{{{s}}}", params[f"{term}_strength"].value)
                L = add_coefficient(rf"\Lambda_{{{s}}}", params[f"{term}_lambda0_nm"].value, r"\mathrm{nm}")
                transparent_terms.append(rf"\frac{{{S}\lambda^2}}{{\lambda^2-{L}^2}}")
                continue

            if kind not in absorptive_kinds:
                raise ValueError(f"Unsupported physical term {kind!r}.")

            E = r"\frac{hc}{\lambda}"

            if kind == "drude":
                D = add_coefficient(rf"D_{{{s}}}", params[f"{term}_amp"].value, r"\mathrm{eV}^2")
                G = add_coefficient(rf"\Gamma_{{{s}}}", params[f"{term}_gamma"].value, r"\mathrm{eV}")
                absorption_terms.append(
                    rf"\frac{{{D}{G}}}{{\left({E}\right)\left[\left({E}\right)^2+{G}^2\right]}}"
                )

            elif kind in ("lorentz", "harmonic"):
                A = add_coefficient(rf"A_{{{s}}}", params[f"{term}_amp"].value)
                E0 = add_coefficient(rf"E_{{0,{s}}}", params[f"{term}_e0"].value, r"\mathrm{eV}")
                G = add_coefficient(rf"\Gamma_{{{s}}}", params[f"{term}_gamma"].value, r"\mathrm{eV}")
                shift = rf"+\frac{{{G}^2}}{{4}}" if kind == "harmonic" else ""
                a = rf"{E0}^2-\left({E}\right)^2{shift}"
                absorption_terms.append(
                    rf"\frac{{{A}{G}^2{E0}\left({E}\right)}}{{\left({a}\right)^2+{G}^2\left({E}\right)^2}}"
                )

            elif kind == "gaussian":
                A = add_coefficient(rf"A_{{{s}}}", params[f"{term}_amp"].value)
                E0 = add_coefficient(rf"E_{{0,{s}}}", params[f"{term}_e0"].value, r"\mathrm{eV}")
                G = add_coefficient(rf"\Gamma_{{{s}}}", params[f"{term}_gamma"].value, r"\mathrm{eV}")
                absorption_terms.append(
                    rf"{A}\left["
                    rf"\exp\!\left(-\left(\frac{{2\sqrt{{\ln 2}}\left[{E}-{E0}\right]}}{{{G}}}\right)^2\right)"
                    rf"-\exp\!\left(-\left(\frac{{2\sqrt{{\ln 2}}\left[{E}+{E0}\right]}}{{{G}}}\right)^2\right)"
                    rf"\right]"
                )
                present_numerical_kk.append(kind)

            elif kind == "tauc_lorentz":
                A = add_coefficient(rf"A_{{{s}}}", params[f"{term}_amp"].value, r"\mathrm{eV}")
                Eg = add_coefficient(rf"E_{{g,{s}}}", params[f"{term}_eg"].value, r"\mathrm{eV}")
                E0_value = (params[f"{term}_eg"].value + params[f"{term}_de"].value)
                E0 = add_coefficient(rf"E_{{0,{s}}}", E0_value, r"\mathrm{eV}")
                G = add_coefficient(rf"\Gamma_{{{s}}}", params[f"{term}_gamma"].value, r"\mathrm{eV}")
                absorption_terms.append(
                    rf"\begin{{cases}}"
                    rf"\dfrac{{{A}{G}{E0}\left[{E}-{Eg}\right]^2}}"
                    rf"{{\left({E}\right)\left(\left[\left({E}\right)^2-{E0}^2\right]^2+{G}^2\left({E}\right)^2\right)}},"
                    rf"&{E}>{Eg}\\[1mm]0,&{E}\le {Eg}\end{{cases}}"
                )
                present_numerical_kk.append(kind)

            elif kind == "cody_lorentz":
                A = add_coefficient(rf"A_{{{s}}}", params[f"{term}_amp"].value, r"\mathrm{eV}")
                Eg = add_coefficient(rf"E_{{g,{s}}}", params[f"{term}_eg"].value, r"\mathrm{eV}")
                E0_value = (params[f"{term}_eg"].value + params[f"{term}_de"].value)
                E0 = add_coefficient(rf"E_{{0,{s}}}", E0_value, r"\mathrm{eV}")
                G = add_coefficient(rf"\Gamma_{{{s}}}", params[f"{term}_gamma"].value, r"\mathrm{eV}")
                Ep = add_coefficient(rf"E_{{p,{s}}}", params[f"{term}_ep"].value, r"\mathrm{eV}")
                absorption_terms.append(
                    rf"\begin{{cases}}"
                    rf"\dfrac{{\left[{E}-{Eg}\right]^2}}{{\left[{E}-{Eg}\right]^2+{Ep}^2}}"
                    rf"\dfrac{{{A}{E0}{G}\left({E}\right)}}"
                    rf"{{\left[\left({E}\right)^2-{E0}^2\right]^2+{G}^2\left({E}\right)^2}},"
                    rf"&{E}>{Eg}\\[1mm]0,&{E}\le {Eg}\end{{cases}}"
                )
                present_numerical_kk.append(kind)

        has_absorption = bool(absorption_terms)
        n_symbol = rf"n_{{{subscript}}}" if subscript else "n"
        eps_imag_symbol = rf"\varepsilon_{{{subscript}}}^{{\prime\prime}}" if subscript else r"\varepsilon^{\prime\prime}"

        transparent_eps_expression = transparent_terms[0] + "".join("+" + piece for piece in transparent_terms[1:])

        if not has_absorption:
            if model.background == "cauchy" and len(transparent_terms) == 1:
                equation = rf"{n_symbol}(\lambda)={cauchy_n}"
            else:
                equation = rf"{n_symbol}(\lambda)=\sqrt{{{transparent_eps_expression}}}"
            real_equation = None
            quantity = "n"
            kk_note = None
        else:
            epsilon_imag_expression = absorption_terms[0] + "".join("+" + piece for piece in absorption_terms[1:])
            equation = rf"{eps_imag_symbol}(\lambda)={epsilon_imag_expression}"
            eps_real_symbol = rf"\varepsilon_{{{subscript}}}^{{\prime}}" if subscript else r"\varepsilon^{\prime}"
            real_equation = (
                rf"{eps_real_symbol}(\lambda)={transparent_eps_expression}"
                rf"+\mathcal{{K}}\!\left[{eps_imag_symbol}(\lambda)\right]"
            )
            quantity = "epsilon_imag"
            kk_note_parts = [
                r"Here $\mathcal{K}$ denotes the Kramers--Kronig transform that gives the dispersive contribution to $\varepsilon'(\lambda)$ from $\varepsilon''(\lambda)$."
            ]
            if present_numerical_kk:
                pretty = {
                    "gaussian": "Gaussian",
                    "tauc_lorentz": "Tauc--Lorentz",
                    "cody_lorentz": "Cody--Lorentz",
                }
                kinds = []
                for kind in present_numerical_kk:
                    name = pretty[kind]
                    if name not in kinds:
                        kinds.append(name)
                kk_note_parts.append(
                    f"For {'/'.join(kinds)} contributions, this transform is evaluated numerically in the fitting code."
                )
            kk_note = " ".join(kk_note_parts)

        coefficient_latex = r"\begin{aligned}" + r"\\".join(
            rf"{name}&={fmt(value)}\,{unit}" if unit else rf"{name}&={fmt(value)}"
            for name, value, unit in coefficients
        ) + r"\end{aligned}"

        reports[direction] = {
            "equation_latex": equation,
            "real_equation_latex": real_equation,
            "quantity": quantity,
            "model_name": _physical_model_display_name(model),
            "coefficients_latex": coefficient_latex,
            "coefficients": coefficients,
            "space": "wavelength",
            "wavelength_unit": "m",
            "hc_ev_m": float(model.hc_ev_nm) * 1e-9,
            "kk_note": kk_note,
        }

    return reports


def print_physical_fit_report(fit_result, model, stack_model, precision=6):
    """Display a report-ready physical model with explicit coefficient units."""
    reports = physical_fit_latex(fit_result, model, stack_model, precision=precision)
    labels = {"iso": "isotropic", "o": "parallel", "e": "perpendicular"}
    try:
        from IPython.display import display, Markdown, Math
    except ImportError:
        display = Markdown = Math = None

    for direction, report in reports.items():
        if report["quantity"] == "n":
            title = f"Final {labels[direction]} refractive-index model"
        else:
            title = f"Final {labels[direction]} absorptive dielectric model"

        if display is not None:
            display(Markdown(f"**{title}:** {report['model_name']}"))
            if report["quantity"] == "epsilon_imag":
                display(Markdown("**Imaginary dielectric function:**"))
                display(Math(report["equation_latex"]))
                display(Markdown("**Real dielectric function:**"))
                display(Math(report["real_equation_latex"]))
            else:
                display(Math(report["equation_latex"]))
            display(Markdown(
                rf"*Here, $\lambda$ denotes the physical wavelength; no unit is encoded in the variable name. "
                rf"Each dimensional coefficient carries its unit explicitly. For $hc/\lambda$, "
                rf"$hc={report['hc_ev_m']:.8g}\,\mathrm{{eV\,m}}$, so $\lambda$ in metres gives energy in eV. "
                rf"When evaluating with plain numbers, convert length-valued coefficients to the same length unit as $\lambda$.*"
            ))
            if report["kk_note"]:
                display(Markdown(f"*{report['kk_note']}*"))
            display(Markdown("**Coefficients:**"))
            display(Math(report["coefficients_latex"]))
        else:
            print(f"\n{title}: {report['model_name']}")
            if report["quantity"] == "epsilon_imag":
                print("  Imaginary dielectric function:", report["equation_latex"])
                print("  Real dielectric function:", report["real_equation_latex"])
            else:
                print("  Formula:", report["equation_latex"])
            print(
                f"  lambda denotes physical wavelength; hc={report['hc_ev_m']:.8g} eV m. "
                "Dimensional coefficient units are written explicitly; convert length quantities "
                "to a common unit for numerical substitution."
            )
            if report["kk_note"]:
                print("  Note:", report["kk_note"])
            print("  Coefficients:")
            for name, value, unit in report["coefficients"]:
                plain_unit = (
                    unit.replace(r"\mathrm{eV}", "eV")
                    .replace(r"\mathrm{nm}", "nm")
                    .replace(r"\mu\mathrm{m}", "um")
                    .replace(r"\mathrm{m}", "m")
                )
                print(f"    {name} = {float(value):.{int(precision)}g}" + (f" {plain_unit}" if plain_unit else ""))
    return reports

def passive_index(eps):
    N = np.sqrt(np.asarray(eps,dtype=complex))
    return np.where((N.imag < 0) | ((np.abs(N.imag) < 1e-14) & (N.real < 0)),-N,N)

def _nk_output_vector(parameter_set,optical_model,stack_model):
    eps_parallel,eps_perp = dielectric_from_model(parameter_set,optical_model,stack_model)
    N_parallel,N_perp = passive_index(eps_parallel),passive_index(eps_perp)
    if stack_model.is_isotropic: return np.concatenate([N_parallel.real,N_parallel.imag])
    return np.concatenate([N_parallel.real,N_parallel.imag,N_perp.real,N_perp.imag])

def _output_jacobian(parameter_set, var_names, optical_model, stack_model):
    y0 = _nk_output_vector(parameter_set,optical_model,stack_model)
    J = np.empty((y0.size,len(var_names)),dtype=float)
    for j,name in enumerate(var_names):
        par = parameter_set[name]
        x0 = float(par.value)
        h = 1e-5*max(abs(x0),1.0)
        if par.stderr is not None and np.isfinite(par.stderr) and par.stderr > 0: h = max(h,1e-3*float(par.stderr))
        lo,hi = float(par.min),float(par.max)
        h_plus,h_minus = h,h
        if np.isfinite(hi): h_plus = min(h_plus,max(hi-x0,0.0))
        if np.isfinite(lo): h_minus = min(h_minus,max(x0-lo,0.0))
        if h_plus > 0 and h_minus > 0:
            p_plus,p_minus = parameter_set.copy(),parameter_set.copy()
            p_plus[name].set(value=x0+h_plus); p_minus[name].set(value=x0-h_minus)
            J[:,j] = (_nk_output_vector(p_plus,optical_model,stack_model)-_nk_output_vector(p_minus,optical_model,stack_model))/(h_plus+h_minus)
        elif h_plus > 0:
            p_plus = parameter_set.copy(); p_plus[name].set(value=x0+h_plus)
            J[:,j] = (_nk_output_vector(p_plus,optical_model,stack_model)-y0)/h_plus
        elif h_minus > 0:
            p_minus = parameter_set.copy(); p_minus[name].set(value=x0-h_minus)
            J[:,j] = (y0-_nk_output_vector(p_minus,optical_model,stack_model))/h_minus
        else: J[:,j] = 0.0
    return J

def refractive_index_with_uncertainty(fit_result,optical_model,stack_model):
    """Convert the reconstructed dielectric function to n,k and propagate the lmfit covariance matrix."""
    if not stack_model.has_unknown: raise ValueError("Loaded optical constants are fixed and therefore do not have fit-derived n,k uncertainties.")
    eps_parallel,eps_perp = dielectric_from_model(fit_result.params,optical_model,stack_model)
    N_parallel,N_perp = passive_index(eps_parallel),passive_index(eps_perp)
    n_wavelengths = len(optical_model.wl_nm)
    if getattr(fit_result,"covar",None) is not None:
        var_names = list(fit_result.var_names)
        J_nk = _output_jacobian(fit_result.params,var_names,optical_model,stack_model)
        cov_params = np.asarray(fit_result.covar,dtype=float)
        nk_variance = np.einsum("ij,jk,ik->i",J_nk,cov_params,J_nk)
        nk_sigma = np.sqrt(np.clip(nk_variance,0.0,None))
        uncertainty_available = True
    else:
        nk_sigma = None
        uncertainty_available = False
    if stack_model.is_isotropic:
        sigma_n = nk_sigma[:n_wavelengths] if nk_sigma is not None else np.full(n_wavelengths,np.nan)
        sigma_k = nk_sigma[n_wavelengths:2*n_wavelengths] if nk_sigma is not None else np.full(n_wavelengths,np.nan)
        return {"epsilon":eps_parallel,"n":N_parallel.real,"k":N_parallel.imag,"sigma_n":sigma_n,"sigma_k":sigma_k,"eps_parallel":eps_parallel,"eps_perp":eps_parallel,"n_parallel":N_parallel.real,"k_parallel":N_parallel.imag,"n_perp":N_parallel.real,"k_perp":N_parallel.imag,"sigma_n_parallel":sigma_n,"sigma_k_parallel":sigma_k,"sigma_n_perp":sigma_n,"sigma_k_perp":sigma_k,"uncertainty_available":uncertainty_available}
    if nk_sigma is not None:
        sigma_n_parallel = nk_sigma[:n_wavelengths]
        sigma_k_parallel = nk_sigma[n_wavelengths:2*n_wavelengths]
        sigma_n_perp = nk_sigma[2*n_wavelengths:3*n_wavelengths]
        sigma_k_perp = nk_sigma[3*n_wavelengths:4*n_wavelengths]
    else:
        sigma_n_parallel,sigma_k_parallel,sigma_n_perp,sigma_k_perp = [np.full(n_wavelengths,np.nan) for _ in range(4)]
    return {"eps_parallel":eps_parallel,"eps_perp":eps_perp,"n_parallel":N_parallel.real,"k_parallel":N_parallel.imag,"n_perp":N_perp.real,"k_perp":N_perp.imag,"sigma_n_parallel":sigma_n_parallel,"sigma_k_parallel":sigma_k_parallel,"sigma_n_perp":sigma_n_perp,"sigma_k_perp":sigma_k_perp,"uncertainty_available":uncertainty_available}



def thickness_fit_table(fit_result, stack_model):
    """Return the final thickness of every finite layer in a single-sample fit.

    Both fitted and fixed thicknesses are included so the table is a complete
    description of the final stack geometry used by the forward model.
    Thicknesses are reported in nm because that is the internal lmfit thickness unit.
    """
    rows = []
    for name in stack_model.thickness_defs:
        parameter_name = thickness_parameter_name(stack_model,name)
        par = fit_result.params[parameter_name]
        rows.append({
            "layer":name,
            "thickness_nm":float(par.value),
            "stderr_nm":np.nan if par.stderr is None else float(par.stderr),
            "fitted":bool(par.vary),
        })
    return pd.DataFrame(rows)
