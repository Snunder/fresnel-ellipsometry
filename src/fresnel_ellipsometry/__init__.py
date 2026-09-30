"""Fresnel ellipsometry fitting for isotropic and optic-axis-normal uniaxial films.

Only the supported workflow API is re-exported here. Numerical implementation
helpers remain private to their modules so the public surface stays small enough
to maintain and document.
"""

from ._version import __version__
from .models import (
    MeasurementData,
    MultiFitDataset,
    PhysicalModel,
    SplineModel,
    StackModel,
    configure_multifit_thickness_parameters,
    thickness_parameter_name,
)
from .io import (
    build_measurement_data,
    experimental_wavelength_grid,
    interpolate_dielectric_function,
    measurement_bandwidth,
    read_ellipsometry_data,
    read_material_data,
    smooth_and_interpolate_exp,
    spectral_quadrature_weights,
)
from .dielectric import (
    add_physical_term,
    build_fit_parameters,
    build_initial_physical_model,
    build_initial_spline,
    build_physical_parameters,
    build_stack_model,
    build_thickness_parameters,
)
from .kramers_kronig import im2re
from .fresnel import fresnel_reflection_all, wrap_delta_deg
from .fitting import fit_physical, fit_spline, fit_thicknesses
from .adaptive_spline import adaptive_fit_spline
from .adaptive_physical import adaptive_fit_physical
from .multifit import (
    build_multi_fit_parameters,
    build_multi_physical_parameters,
    build_multi_thickness_parameters,
    fit_physical_multi,
    fit_spline_multi,
    fit_thicknesses_multi,
    multifit_backend_info,
    multi_dataset_fit_table,
)
from .adaptive_multi import adaptive_fit_physical_multi, adaptive_fit_spline_multi
from .prefit import (
    automatic_thickness_prefit,
    clear_thickness_anchors,
    print_thickness_prefit_summary,
    seed_fitted_thicknesses,
    set_dielectric_guard,
    set_thickness_anchors,
    varying_thickness_names,
)
from .reporting import (
    passive_index,
    physical_fit_latex,
    print_physical_fit_report,
    refractive_index_with_uncertainty,
    thickness_fit_table,
)
from .plotting import (
    plot_fit,
    plot_loaded_data,
    plot_loaded_datasets,
    plot_refractive_index,
    plot_residual_evolution,
)
from .export import save_refractive_index_csv

__all__ = [
    "__version__",
    # Data models and naming.
    "MeasurementData",
    "MultiFitDataset",
    "PhysicalModel",
    "SplineModel",
    "StackModel",
    "configure_multifit_thickness_parameters",
    "thickness_parameter_name",
    # I/O and preprocessing.
    "read_ellipsometry_data",
    "read_material_data",
    "experimental_wavelength_grid",
    "smooth_and_interpolate_exp",
    "interpolate_dielectric_function",
    "measurement_bandwidth",
    "spectral_quadrature_weights",
    "build_measurement_data",
    # Stack and dielectric models.
    "build_stack_model",
    "build_initial_spline",
    "build_initial_physical_model",
    "add_physical_term",
    "build_thickness_parameters",
    "build_fit_parameters",
    "build_physical_parameters",
    # Forward model and direct fits.
    "im2re",
    "fresnel_reflection_all",
    "wrap_delta_deg",
    "fit_thicknesses",
    "fit_spline",
    "fit_physical",
    # Adaptive single-sample fits.
    "adaptive_fit_spline",
    "adaptive_fit_physical",
    # Multi-sample construction and fitting.
    "build_multi_thickness_parameters",
    "build_multi_fit_parameters",
    "build_multi_physical_parameters",
    "fit_thicknesses_multi",
    "fit_spline_multi",
    "fit_physical_multi",
    "adaptive_fit_spline_multi",
    "adaptive_fit_physical_multi",
    "multifit_backend_info",
    "multi_dataset_fit_table",
    # Thickness initialization and optional regularization.
    "varying_thickness_names",
    "seed_fitted_thicknesses",
    "set_thickness_anchors",
    "clear_thickness_anchors",
    "set_dielectric_guard",
    "automatic_thickness_prefit",
    "print_thickness_prefit_summary",
    # Reporting, plotting, and export.
    "passive_index",
    "physical_fit_latex",
    "print_physical_fit_report",
    "refractive_index_with_uncertainty",
    "thickness_fit_table",
    "plot_fit",
    "plot_loaded_data",
    "plot_loaded_datasets",
    "plot_residual_evolution",
    "plot_refractive_index",
    "save_refractive_index_csv",
]
