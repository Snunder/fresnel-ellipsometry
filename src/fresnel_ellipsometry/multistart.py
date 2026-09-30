"""Structured coarse multistart for the initial KK B-spline fit.

The strategy is inspired by the coarse multistart stage used by Gilliot
(Applied Optics 56, 1173--1182, 2017): solve the low-flexibility problem
from several initial conditions before adding local spline complexity.
Here the starts are deterministic and structured rather than purely random,
so the extra cost remains modest and the tested branches are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import thickness_parameter_name
from .fitting import _fit_measurement_chisqr, fit_spline
from .multifit import fit_spline_multi


def _clip_parameter(parameter, value):
    value = float(value)
    lo = float(parameter.min)
    hi = float(parameter.max)
    if np.isfinite(lo):
        value = max(value, lo)
    if np.isfinite(hi):
        value = min(value, hi)
    # Stay infinitesimally inside a finite interval when possible.  This avoids
    # starting a least-squares transform exactly at a hard boundary.
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        margin = 1e-9 * max(abs(lo), abs(hi), 1.0)
        value = min(max(value, lo + margin), hi - margin)
    return value


def _set_value(params, name, value):
    if name in params and getattr(params[name], "vary", True):
        params[name].set(value=_clip_parameter(params[name], value))


def _spline_parameter_names(params, stack_model):
    unknown = stack_model.unknown_name
    if stack_model.is_isotropic:
        offsets = [f"{unknown}_eps_offset"]
        coeff_o = [name for name in params if name.startswith(f"{unknown}_eps2_")]
        return offsets, sorted(coeff_o), []
    offsets = [f"{unknown}_eps_offset_o", f"{unknown}_eps_offset_e"]
    coeff_o = sorted(name for name in params if name.startswith(f"{unknown}_eps2_o_"))
    coeff_e = sorted(name for name in params if name.startswith(f"{unknown}_eps2_e_"))
    return offsets, coeff_o, coeff_e


def _thickness_parameter_names(stack_models):
    names = []
    for stack_model in stack_models:
        for layer, spec in stack_model.thickness_defs.items():
            if spec.get("vary", False):
                name = thickness_parameter_name(stack_model, layer)
                if name not in names:
                    names.append(name)
    return names


def _parameter_signature(params):
    return tuple((name, round(float(par.value), 12)) for name, par in params.items())


def structured_spline_starts(params, stack_model, stack_models=None, n_starts=10):
    """Return deterministic coarse starts spanning dielectric and thickness branches.

    The baseline (including a thickness pre-fit seed, if present) is always first.
    Additional starts vary broad absorption, thickness, and dielectric offsets.  For
    uniaxial materials the set explicitly includes symmetric, parallel-heavy,
    perpendicular-heavy, and swapped anisotropy starts.
    """
    n_starts = max(1, int(n_starts))
    stack_models = [stack_model] if stack_models is None else list(stack_models)
    offsets, coeff_o, coeff_e = _spline_parameter_names(params, stack_model)
    thickness_names = _thickness_parameter_names(stack_models)

    candidates = [("baseline", params.copy())]

    def absorption_start(label, value, direction="both"):
        p = params.copy()
        if direction in ("both", "o"):
            for name in coeff_o:
                _set_value(p, name, value)
        if direction in ("both", "e"):
            for name in coeff_e:
                _set_value(p, name, value)
        candidates.append((label, p))

    # Broad absorption levels cover the common branch where a coarse fit remains
    # pinned close to epsilon_2 = 0 from a transparent initial guess.
    absorption_start("low absorption", 1e-4)
    absorption_start("moderate absorption", 5e-2)

    # Thickness is strongly correlated with dielectric response.  Explore modest
    # perturbations around the pre-fit/current seed rather than arbitrary bounds.
    for factor, label in ((0.85, "thickness -15%"), (1.15, "thickness +15%")):
        if thickness_names:
            p = params.copy()
            for name in thickness_names:
                par = p[name]
                value = float(par.value)
                if abs(value) < 1e-12 and np.isfinite(par.min) and np.isfinite(par.max):
                    value = float(par.min) + (0.35 if factor < 1 else 0.65) * (float(par.max) - float(par.min))
                else:
                    value *= factor
                _set_value(p, name, value)
            candidates.append((label, p))

    if stack_model.is_isotropic:
        offset_name = offsets[0]
        for value in (2.0, 3.0, 4.0):
            p = params.copy()
            _set_value(p, offset_name, value)
            for name in coeff_o:
                _set_value(p, name, max(float(p[name].value), 2e-2))
            candidates.append((f"epsilon offset {value:g}", p))
    else:
        o_name, e_name = offsets
        for o_value, e_value, label in (
            (3.0, 3.0, "symmetric epsilon (3,3)"),
            (2.0, 4.0, "epsilon (parallel,perp)=(2,4)"),
            (4.0, 2.0, "epsilon (parallel,perp)=(4,2)"),
        ):
            p = params.copy()
            _set_value(p, o_name, o_value)
            _set_value(p, e_name, e_value)
            for name in coeff_o + coeff_e:
                _set_value(p, name, max(float(p[name].value), 2e-2))
            candidates.append((label, p))

        # Swap the complete o/e initialization.  This is useful when both tensor
        # directions can initially trade off against thickness.
        p = params.copy()
        o_offset, e_offset = float(params[o_name].value), float(params[e_name].value)
        _set_value(p, o_name, e_offset)
        _set_value(p, e_name, o_offset)
        for left, right in zip(coeff_o, coeff_e):
            left_value, right_value = float(params[left].value), float(params[right].value)
            _set_value(p, left, right_value)
            _set_value(p, right, left_value)
        candidates.append(("swap parallel/perpendicular seed", p))

    absorption_start("strong absorption", 2.5e-1)
    if stack_model.is_uniaxial:
        absorption_start("parallel absorption", 1.5e-1, direction="o")
        absorption_start("perpendicular absorption", 1.5e-1, direction="e")

    # Remove accidental duplicates caused by bounds or equal initial values.
    unique = []
    seen = set()
    for label, candidate in candidates:
        signature = _parameter_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append((label, candidate))
        if len(unique) >= n_starts:
            break
    return unique


def _history_row(stage, rank, label, result):
    return {
        "stage": stage,
        "rank": int(rank),
        "start": label,
        "chisqr": _fit_measurement_chisqr(result),
        "sigma_rms": float(getattr(result, "sigma_rms", np.nan)),
        "success": bool(getattr(result, "success", False)),
        "nfev": int(getattr(result, "nfev", -1)),
    }


def _coarse_multistart(starts, fit_function, screen_max_nfev, max_nfev, n_finalists, show_progress, verbose):
    screened = []
    rows = []
    n = len(starts)
    for i, (label, start) in enumerate(starts, 1):
        if show_progress:
            print(f"Coarse multistart screen {i}/{n}", end="\r", flush=True)
        result = fit_function(start, max_nfev=int(screen_max_nfev))
        chisqr = _fit_measurement_chisqr(result)
        if np.isfinite(chisqr):
            screened.append((label, result))
        rows.append(_history_row("screen", i, label, result))
        if verbose:
            print(f"screen {i:02d} {label}: RMS={getattr(result, 'sigma_rms', np.nan):.3f} sigma")
    if not screened:
        raise RuntimeError("Every coarse multistart screen returned a non-finite measurement chi-square.")

    screened.sort(key=lambda item: _fit_measurement_chisqr(item[1]))
    finalists = screened[:max(1, min(int(n_finalists), len(screened)))]
    converged = []
    for i, (label, screen_result) in enumerate(finalists, 1):
        if show_progress:
            print(f"Coarse multistart finalist {i}/{len(finalists)}", end="\r", flush=True)
        result = fit_function(screen_result.params, max_nfev=int(max_nfev))
        chisqr = _fit_measurement_chisqr(result)
        if np.isfinite(chisqr):
            converged.append((label, result))
        rows.append(_history_row("final", i, label, result))
        if verbose:
            print(f"final {i:02d} {label}: RMS={getattr(result, 'sigma_rms', np.nan):.3f} sigma")
    if show_progress:
        print(" " * 70, end="\r", flush=True)
    if not converged:
        raise RuntimeError("Every coarse multistart finalist returned a non-finite measurement chi-square.")
    best_label, best_result = min(converged, key=lambda item: _fit_measurement_chisqr(item[1]))
    history = pd.DataFrame(rows)
    best_result.initial_multistart_history = history
    best_result.initial_multistart_label = best_label
    if show_progress:
        print(f"Coarse multistart selected: {best_label} | RMS={best_result.sigma_rms:.3f} sigma")
    return best_result, history


def coarse_multistart_spline(
    params, spline, stack_model, measurement,
    n_starts=10, screen_max_nfev=250, n_finalists=3,
    max_nfev=100000, show_progress=True, verbose=False,
):
    """Robustly initialize a single-sample coarse spline fit."""
    starts = structured_spline_starts(params, stack_model, n_starts=n_starts)
    return _coarse_multistart(
        starts,
        lambda p, max_nfev: fit_spline(p, spline, stack_model, measurement, max_nfev=max_nfev),
        screen_max_nfev, max_nfev, n_finalists, show_progress, verbose,
    )


def coarse_multistart_spline_multi(
    params, spline, datasets,
    n_starts=10, screen_max_nfev=250, n_finalists=3,
    max_nfev=100000, show_progress=True, verbose=False,
):
    """Robustly initialize one coarse dielectric tensor shared by several samples."""
    datasets = list(datasets)
    if not datasets:
        raise ValueError("datasets may not be empty.")
    reference = datasets[0].stack_model
    starts = structured_spline_starts(
        params, reference,
        stack_models=[dataset.stack_model for dataset in datasets],
        n_starts=n_starts,
    )
    return _coarse_multistart(
        starts,
        lambda p, max_nfev: fit_spline_multi(p, spline, datasets, max_nfev=max_nfev),
        screen_max_nfev, max_nfev, n_finalists, show_progress, verbose,
    )
