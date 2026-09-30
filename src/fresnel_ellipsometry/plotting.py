"""Plotting helpers with the formatting used in the original research notebooks."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .fresnel import fresnel_reflection_all


AOI_COLOR_START = "#C7CEFF"
AOI_COLOR_END = "#001AFF"
ITER_COLOR_START = "#D9DEFF"
ITER_COLOR_END = "#001AFF"
PARALLEL_COLOR = "#001AFF"
PERPENDICULAR_COLOR = "#FF0000"


def _aoi_colors(aois, name="aoi"):
    """Return the historical pale-blue -> blue AOI color mapping."""
    aois = tuple(aois)
    cmap = LinearSegmentedColormap.from_list(
        str(name), [AOI_COLOR_START, AOI_COLOR_END], N=max(len(aois), 2)
    )
    denominator = max(len(aois) - 1, 1)
    return {aoi: cmap(i / denominator) for i, aoi in enumerate(aois)}


def plot_loaded_data(meas, aois, label=None, figsize=(12, 4)):
    """Plot the raw and processed ellipsometry data before fitting.

    The raw native samples are always shown as points.  If interpolation and/or
    smoothing was requested, the actual arrays passed to the fit are overlaid as
    lines.  This makes it easy to verify preprocessing before starting an inverse
    problem.
    """
    aois = tuple(aois)
    colors = _aoi_colors(aois, name=f"aoi_{str(label).replace(' ', '_')}")
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    for aoi in aois:
        item = meas[aoi]
        color = colors[aoi]
        raw_wl = np.asarray(item.get("raw_wl_nm"), dtype=float)
        raw_psi = np.rad2deg(np.asarray(item.get("psi_rad_raw"), dtype=float))
        raw_delta = np.mod(np.rad2deg(np.asarray(item.get("delta_rad_raw"), dtype=float)), 360.0)

        processed_wl = np.asarray(item.get("target_wl_nm", item.get("wl_nm", [])), dtype=float)
        if processed_wl.size == 0:
            # smooth_and_interpolate_exp historically did not store the target grid,
            # so infer it from the processed array length when the caller did not add it.
            processed_wl = raw_wl if len(item["psi_rad"]) == len(raw_wl) else np.linspace(raw_wl.min(), raw_wl.max(), len(item["psi_rad"]))

        has_smoothing = "psi_rad_smooth" in item and "delta_rad_smooth" in item
        has_interpolation = bool(item.get("spline_interpolated", False))
        show_processed = has_smoothing or has_interpolation

        raw_alpha = 0.45 if show_processed else 0.95
        axes[0].plot(raw_wl, raw_psi, ".", color=color, ms=4, alpha=raw_alpha, label=fr"{aoi:g}$^\circ$ raw")
        axes[1].plot(raw_wl, raw_delta, ".", color=color, ms=4, alpha=raw_alpha, label=fr"{aoi:g}$^\circ$ raw")

        if show_processed:
            psi = np.rad2deg(item["psi_rad_smooth"] if has_smoothing else item["psi_rad"])
            delta = np.mod(np.rad2deg(item["delta_rad_smooth"] if has_smoothing else item["delta_rad"]), 360.0)
            linestyle = "--" if has_smoothing else "-"
            tag_parts = []
            if has_smoothing:
                tag_parts.append("smoothed")
            if has_interpolation:
                tag_parts.append("interpolated")
            tag = "+".join(tag_parts) if tag_parts else "processed"
            axes[0].plot(processed_wl, psi, linestyle, color=color, lw=2, alpha=0.9, label=fr"{aoi:g}$^\circ$ {tag}")
            axes[1].plot(processed_wl, delta, linestyle, color=color, lw=2, alpha=0.9, label=fr"{aoi:g}$^\circ$ {tag}")

    axes[0].set(xlabel=r"$\lambda$ (nm)", ylabel=r"$\Psi$ (deg)")
    axes[1].set(xlabel=r"$\lambda$ (nm)", ylabel=r"$\Delta$ (deg)")
    for ax in axes:
        ax.legend(frameon=False)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    if label:
        fig.suptitle(f"Loaded ellipsometry: {label}")
    return fig, axes



def plot_loaded_datasets(loaded, figsize=None):
    """Plot preprocessing for several datasets in the original multi-sample layout.

    ``loaded`` maps a display label to a dictionary containing ``meas``, ``AOIs``
    and ``wl_nm``. Raw native points are overlaid whenever interpolation or
    smoothing changes what is passed to the fit.
    """
    items = list(loaded.items())
    if not items:
        raise ValueError("loaded may not be empty.")
    if figsize is None:
        figsize = (11, max(3, 2.8 * len(items)))
    fig, axes = plt.subplots(len(items), 2, figsize=figsize, squeeze=False, constrained_layout=True)
    for row, (label, item) in enumerate(items):
        aois = tuple(item["AOIs"])
        wl_nm = np.asarray(item["wl_nm"], dtype=float)
        colors = _aoi_colors(aois, name=f"aoi_{row}")
        for aoi in aois:
            m = item["meas"][aoi]
            color = colors[aoi]
            interpolated = bool(m.get("spline_interpolated", False))
            smoothed = "psi_rad_smooth" in m and "delta_rad_smooth" in m
            changed = interpolated or smoothed

            raw_wl = np.asarray(m["raw_wl_nm"], dtype=float)
            raw_psi = np.rad2deg(np.asarray(m["psi_rad_raw"], dtype=float))
            raw_delta = np.mod(np.rad2deg(np.asarray(m["delta_rad_raw"], dtype=float)), 360.0)

            if changed:
                axes[row,0].plot(raw_wl, raw_psi, ".", color=color, ms=4, alpha=0.55, label=f"{aoi:g}° raw")
                axes[row,1].plot(raw_wl, raw_delta, ".", color=color, ms=4, alpha=0.55, label=f"{aoi:g}° raw")
                if interpolated:
                    axes[row,0].plot(wl_nm, np.rad2deg(m["psi_rad"]), color=color, lw=2, alpha=0.60, label=f"{aoi:g}° PCHIP")
                    axes[row,1].plot(wl_nm, np.mod(np.rad2deg(m["delta_rad"]),360.0), color=color, lw=2, alpha=0.60, label=f"{aoi:g}° PCHIP")
                if smoothed:
                    axes[row,0].plot(wl_nm, np.rad2deg(m["psi_rad_smooth"]), "--", color=color, lw=2, alpha=0.85, label=f"{aoi:g}° smoothed")
                    axes[row,1].plot(wl_nm, np.mod(np.rad2deg(m["delta_rad_smooth"]),360.0), "--", color=color, lw=2, alpha=0.85, label=f"{aoi:g}° smoothed")
            else:
                axes[row,0].plot(wl_nm, np.rad2deg(m["psi_rad"]), ".", color=color, ms=5, label=f"{aoi:g}°")
                axes[row,1].plot(wl_nm, np.mod(np.rad2deg(m["delta_rad"]),360.0), ".", color=color, ms=5, label=f"{aoi:g}°")

        axes[row,0].set_ylabel(rf"{label}  $\Psi$ (deg)")
        axes[row,1].set_ylabel(rf"{label}  $\Delta$ (deg)")
        axes[row,0].legend(frameon=False)
        for ax in axes[row]:
            ax.set_xlim(wl_nm.min(), wl_nm.max())
            ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (nm)")
    return fig, axes

def plot_fit(fit_result, datasets, optical_model=None):
    """Plot measured and fitted Psi/Delta using the original blue AOI gradient."""
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
    fig, axes = plt.subplots(
        len(datasets), 2,
        figsize=(11, max(3, 3 * len(datasets))),
        squeeze=False,
        constrained_layout=True,
    )
    for row, dataset in enumerate(datasets):
        response = fresnel_reflection_all(
            fit_result.params,
            dataset.measurement.AOIs,
            optical_model,
            dataset.stack_model,
        )
        colors = _aoi_colors(dataset.measurement.AOIs, name=f"fit_{row}")
        for aoi in dataset.measurement.AOIs:
            color = colors[aoi]
            axes[row, 0].plot(
                dataset.measurement.wl_nm, response[aoi]["psi"],
                color=color, lw=2, label=f"{aoi:g}° fit",
            )
            axes[row, 0].errorbar(
                dataset.measurement.wl_nm,
                dataset.measurement.psi_exp_deg[aoi],
                yerr=dataset.measurement.psi_sigma_deg[aoi],
                fmt=".", markersize=3, color=color, alpha=0.4,
            )
            axes[row, 1].plot(
                dataset.measurement.wl_nm, response[aoi]["delta"],
                color=color, lw=2, label=f"{aoi:g}° fit",
            )
            axes[row, 1].errorbar(
                dataset.measurement.wl_nm,
                dataset.measurement.delta_exp_deg[aoi],
                yerr=dataset.measurement.delta_sigma_deg[aoi],
                fmt=".", markersize=3, color=color, alpha=0.4,
            )
        axes[row, 0].set_ylabel(rf"{dataset.label}  $\Psi$ (deg)")
        axes[row, 1].set_ylabel(rf"{dataset.label}  $\Delta$ (deg)")
        axes[row, 0].legend(frameon=False)
        for ax in axes[row]:
            ax.set_xlim(dataset.measurement.wl_nm.min(), dataset.measurement.wl_nm.max())
            ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (nm)")
    return fig, axes


def plot_residual_evolution(fit_snapshots, datasets, multi_sample=None):
    """Plot signed Psi/Delta residuals using the original iteration colormap."""
    if not fit_snapshots:
        raise ValueError("fit_snapshots is empty.")
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
    if multi_sample is None:
        multi_sample = len(datasets) > 1

    iterations = np.asarray([item["iteration"] for item in fit_snapshots], dtype=float)
    norm = Normalize(vmin=float(iterations.min()), vmax=float(iterations.max()) if iterations.max() > iterations.min() else float(iterations.min() + 1))
    cmap = LinearSegmentedColormap.from_list(
        "fit_iterations", [ITER_COLOR_START, ITER_COLOR_END], N=256
    )
    figures = []
    for dataset in datasets:
        fig, axes = plt.subplots(
            2,
            len(dataset.measurement.AOIs),
            figsize=(max(7, 4 * len(dataset.measurement.AOIs)), 5),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        for item in fit_snapshots:
            stored = item["datasets"][dataset.label] if multi_sample else item
            color = cmap(norm(item["iteration"]))
            for col, aoi in enumerate(dataset.measurement.AOIs):
                axes[0, col].plot(
                    dataset.measurement.wl_nm,
                    stored["psi_residual_deg"][aoi],
                    color=color, lw=1.5, alpha=0.9,
                )
                axes[1, col].plot(
                    dataset.measurement.wl_nm,
                    stored["delta_residual_deg"][aoi],
                    color=color, lw=1.5, alpha=0.9,
                )
        for col, aoi in enumerate(dataset.measurement.AOIs):
            axes[0, col].set_title(fr"{dataset.label}: {aoi:g}$^\circ$")
            axes[1, col].set_xlabel("Wavelength (nm)")
            axes[0, col].axhline(0.0, color="k", lw=1.5, ls="--", alpha=0.5)
            axes[1, col].axhline(0.0, color="k", lw=1.5, ls="--", alpha=0.5)
        axes[0, 0].set_ylabel(r"$\Psi_\mathrm{fit}-\Psi_\mathrm{exp}$ (deg)")
        axes[1, 0].set_ylabel(r"$\Delta_\mathrm{fit}-\Delta_\mathrm{exp}$ (deg)")
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), label="Fit iteration", pad=0.02)
        figures.append((fig, axes))
    return figures


def _finite_uncertainty(index_result, key, n):
    value = index_result.get(key)
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.size != n or not np.any(np.isfinite(value)):
        return None
    return value


def plot_refractive_index(wl_nm, index_result, figsize=(12, 3)):
    """Plot isotropic or uniaxial n/k using the historical blue/red formatting."""
    wl_nm = np.asarray(wl_nm, dtype=float)
    fig, (ax_n, ax_k) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    if "n" in index_result:
        n = np.asarray(index_result["n"], dtype=float)
        k = np.asarray(index_result["k"], dtype=float)
        ax_n.plot(wl_nm, n, label="$n$", lw=2, color=PARALLEL_COLOR, zorder=99)
        ax_k.plot(wl_nm, k, label="$k$", lw=2, color=PARALLEL_COLOR, zorder=99)
        sigma_n = _finite_uncertainty(index_result, "sigma_n", len(wl_nm))
        sigma_k = _finite_uncertainty(index_result, "sigma_k", len(wl_nm))
        if sigma_n is not None:
            ax_n.fill_between(wl_nm, n - sigma_n, n + sigma_n, color=PARALLEL_COLOR, alpha=0.20)
        if sigma_k is not None:
            ax_k.fill_between(wl_nm, k - sigma_k, k + sigma_k, color=PARALLEL_COLOR, alpha=0.20)
    else:
        n_parallel = np.asarray(index_result["n_parallel"], dtype=float)
        n_perp = np.asarray(index_result["n_perp"], dtype=float)
        k_parallel = np.asarray(index_result["k_parallel"], dtype=float)
        k_perp = np.asarray(index_result["k_perp"], dtype=float)
        ax_n.plot(wl_nm, n_parallel, label=r"$n_\parallel$", lw=2, color=PARALLEL_COLOR, zorder=99)
        ax_n.plot(wl_nm, n_perp, label=r"$n_\perp$", lw=2, color=PERPENDICULAR_COLOR, zorder=99)
        ax_k.plot(wl_nm, k_parallel, label=r"$k_\parallel$", lw=2, color=PARALLEL_COLOR, zorder=99)
        ax_k.plot(wl_nm, k_perp, label=r"$k_\perp$", lw=2, color=PERPENDICULAR_COLOR, zorder=99)
        for ax, values, key, color in (
            (ax_n, n_parallel, "sigma_n_parallel", PARALLEL_COLOR),
            (ax_n, n_perp, "sigma_n_perp", PERPENDICULAR_COLOR),
            (ax_k, k_parallel, "sigma_k_parallel", PARALLEL_COLOR),
            (ax_k, k_perp, "sigma_k_perp", PERPENDICULAR_COLOR),
        ):
            sigma = _finite_uncertainty(index_result, key, len(wl_nm))
            if sigma is not None:
                ax.fill_between(wl_nm, values - sigma, values + sigma, color=color, alpha=0.20)

    ax_n.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_n.set_ylabel("n", fontsize=12)
    ax_k.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_k.set_ylabel("k", fontsize=12)
    for ax in (ax_n, ax_k):
        ax.legend(frameon=False)
        ax.set_xlim(wl_nm.min(), wl_nm.max())
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    return fig, (ax_n, ax_k)
