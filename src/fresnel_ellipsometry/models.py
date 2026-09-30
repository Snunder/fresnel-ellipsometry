"""Core data models and shared naming helpers."""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class SplineModel:
    wl_nm: np.ndarray
    E_meas: np.ndarray
    E_kk: np.ndarray
    internal_knots: np.ndarray
    knot_vector: np.ndarray
    B_kk: np.ndarray
    B_meas: np.ndarray
    KK_basis_kk: np.ndarray
    KK_basis_meas: np.ndarray
    degree: int
    kk_e_min: float
    kk_e_max: float
    hc_ev_nm: float
    @property
    def n_free_nodes(self):
        return self.B_kk.shape[1]
    @property
    def knot_wavelengths_nm(self):
        return np.sort(self.hc_ev_nm/self.internal_knots) if len(self.internal_knots) else np.array([])

@dataclass(frozen=True)
class PhysicalModel:
    """A transparent background plus additive dielectric susceptibilities."""
    wl_nm: np.ndarray
    background: str
    terms: tuple
    kk_e_min: float
    kk_e_max: float
    kk_points: int
    hc_ev_nm: float
    @property
    def n_terms(self):
        return len(self.terms)
    @property
    def description(self):
        pieces = ([] if self.background == "constant" else [f"{self.background} background"])+list(self.terms)
        return " + ".join(pieces) if pieces else "constant epsilon"

@dataclass
class StackModel:
    names: list
    unknown_index: object
    unknown_name: object
    known_eps: dict
    thickness_defs: dict
    wl_nm: np.ndarray
    symmetry: str
    known_eps_e: dict = field(default_factory=dict)
    thickness_anchors: dict = field(default_factory=dict)
    dielectric_guard: dict = field(default_factory=dict)
    thickness_param_names: dict = field(default_factory=dict)
    _forward_cache: dict = field(default_factory=dict,repr=False)
    @property
    def has_unknown(self):
        return self.unknown_index is not None
    @property
    def is_isotropic(self):
        return self.symmetry == "isotropic"
    @property
    def is_uniaxial(self):
        return self.symmetry == "uniaxial"

@dataclass
class MeasurementData:
    wl_nm: np.ndarray
    AOIs: tuple
    psi_exp_deg: dict
    delta_exp_deg: dict
    psi_sigma_deg: dict
    delta_sigma_deg: dict
    spectral_weighting: str
    spectral_weights: np.ndarray

@dataclass
class MultiFitDataset:
    """One flake participating in a shared-dielectric multi-sample fit."""
    label: str
    stack_model: StackModel
    measurement: MeasurementData
    bandwidth_nm: np.ndarray

def _safe_dataset_label(label):
    """Return a valid lmfit/Python identifier derived from a user-facing filename or label."""
    text = "".join(ch if str(ch).isalnum() or ch == "_" else "_" for ch in str(label)).strip("_")
    if not text:
        text = "dataset"
    if not (text[0].isalpha() or text[0] == "_"):
        text = f"flake_{text}"
    return text

def thickness_parameter_name(stack_model,name):
    """Return the lmfit name for one thickness; multi-flake fits prefix it by flake."""
    return (getattr(stack_model,"thickness_param_names",{}) or {}).get(name,f"{name}_d")

def configure_multifit_thickness_parameters(datasets):
    """Assign valid independent lmfit thickness names while keeping filenames as display labels."""
    datasets = list(datasets)
    used = set()
    for dataset in datasets:
        base = _safe_dataset_label(dataset.label)
        label = base
        suffix = 2
        while label in used:
            label = f"{base}_{suffix}"
            suffix += 1
        used.add(label)
        dataset.stack_model.thickness_param_names = {
            name:f"{label}__{_safe_dataset_label(name)}_d" for name in dataset.stack_model.thickness_defs
        }
    return datasets

