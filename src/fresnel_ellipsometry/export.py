"""Export helpers for reconstructed optical constants."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def save_refractive_index_csv(
    wl_nm,
    index_result,
    material_name,
    output_dir="results",
    suffix="",
    include_uncertainty=False,
):
    """Save fitted optical constants in the package's reusable ``wl,n,k`` format.

    ``wl`` is written in micrometres, matching :func:`read_material_data` input.
    Isotropic data is saved as ``<material><suffix>.csv``.  Uniaxial data is
    saved as ``<material><suffix>_o.csv`` (parallel/ordinary) and
    ``<material><suffix>_e.csv`` (perpendicular/extraordinary).

    Returns
    -------
    list[pathlib.Path]
        Paths of the files written.
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    material_name = str(material_name).strip()
    if not material_name:
        raise ValueError("material_name may not be empty.")
    suffix = str(suffix)

    def frame(n, k, sigma_n=None, sigma_k=None):
        values = {
            "wl": wl_nm * 1e-3,
            "n": np.asarray(n, dtype=float),
            "k": np.asarray(k, dtype=float),
        }
        if include_uncertainty and sigma_n is not None and sigma_k is not None:
            values["n_sigma"] = np.asarray(sigma_n, dtype=float)
            values["k_sigma"] = np.asarray(sigma_k, dtype=float)
        return pd.DataFrame(values)

    written = []
    if "n" in index_result and "k" in index_result:
        path = output_dir / f"{material_name}{suffix}.csv"
        frame(
            index_result["n"], index_result["k"],
            index_result.get("sigma_n"), index_result.get("sigma_k"),
        ).to_csv(path, index=False)
        written.append(path)
        return written

    required = ("n_parallel", "k_parallel", "n_perp", "k_perp")
    missing = [key for key in required if key not in index_result]
    if missing:
        raise KeyError(f"index_result is missing required field(s): {missing}")

    path_o = output_dir / f"{material_name}{suffix}_o.csv"
    path_e = output_dir / f"{material_name}{suffix}_e.csv"
    frame(
        index_result["n_parallel"], index_result["k_parallel"],
        index_result.get("sigma_n_parallel"), index_result.get("sigma_k_parallel"),
    ).to_csv(path_o, index=False)
    frame(
        index_result["n_perp"], index_result["k_perp"],
        index_result.get("sigma_n_perp"), index_result.get("sigma_k_perp"),
    ).to_csv(path_e, index=False)
    written.extend([path_o, path_e])
    return written
