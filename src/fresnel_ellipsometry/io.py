"""Input, preprocessing, interpolation, and measurement construction."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d

from .models import MeasurementData

def read_ellipsometry_data(path_or_stem):
    path = Path(path_or_stem)
    if path.suffix.lower() in (".csv",".ods"):
        candidates = [path]
    else:
        candidates = [path.with_suffix(".csv"),path.with_suffix(".ods")]
    existing = next((candidate for candidate in candidates if candidate.exists()),None)
    if existing is None: raise FileNotFoundError(f"Could not find any of: {', '.join(str(candidate) for candidate in candidates)}")
    if existing.suffix.lower() == ".csv":
        return pd.read_csv(existing)
    try:
        return pd.read_excel(existing,engine="odf",sheet_name=0)
    except ImportError as exc:
        raise ImportError("Reading .ods files requires the optional Python package 'odfpy'. Install it with: pip install odfpy") from exc

def read_material_data(path_or_stem):
    """Load isotropic "Material" data or a uniaxial "Material_o/e" pair.
    A complete "*_o"/"*_e" pair takes precedence over a same-named
    isotropic file. 
    """
    path = Path(path_or_stem)
    if path.suffix.lower() in (".csv",".ods"):
        return read_ellipsometry_data(path)
    def existing_component(suffix):
        stem = path.with_name(path.name+suffix)
        return next((candidate for candidate in (stem.with_suffix(".csv"),stem.with_suffix(".ods")) if candidate.exists()),None)
    ordinary_path = existing_component("_o")
    extraordinary_path = existing_component("_e")
    if (ordinary_path is None) != (extraordinary_path is None):
        missing = f"{path.name}_{'o' if ordinary_path is None else 'e'}"
        present = ordinary_path if ordinary_path is not None else extraordinary_path
        raise FileNotFoundError(f"Incomplete anisotropic optical data for {path.name}: found {present.name}, but could not find {missing}.csv or {missing}.ods.")
    if ordinary_path is not None:
        return {"o":read_ellipsometry_data(ordinary_path),"e":read_ellipsometry_data(extraordinary_path)}
    return read_ellipsometry_data(path)

def experimental_wavelength_grid(data,AOIs,interpolation=False):
    """Preserve native wavelengths and densify only gaps larger than a chosen spacing.
    "interpolation=False" leaves the reference AOI grid unchanged. A positive
    numeric value such as "interpolation=5" inserts points at 5 nm increments
    only inside larger native gaps. Dense regions and every original wavelength
    are preserved exactly.
    """
    AOIs = tuple(AOIs)
    if len(AOIs) == 0: raise ValueError("At least one AOI is required.")

    if interpolation is False or interpolation is None:
        interpolation_step_nm = None
    elif isinstance(interpolation,(bool,np.bool_)):
        raise ValueError("interpolation must be False or a positive spacing in nm, for example interpolation=5.")
    else:
        interpolation_step_nm = float(interpolation)
        if not np.isfinite(interpolation_step_nm) or interpolation_step_nm <= 0:
            raise ValueError("interpolation must be False or a positive finite spacing in nm.")

    wavelength_grids = []
    for aoi in AOIs:
        x = np.sort(data.loc[np.isclose(data["AOI_(°)"],aoi),"lambda_(nm)"].dropna().unique().astype(float))
        if len(x) < 2: raise ValueError(f"AOI {aoi:g}°: fewer than two unique wavelengths are available.")
        wavelength_grids.append(x)

    # Never extrapolate an AOI: use only the wavelength overlap shared by all angles.
    wl_min = max(float(x[0]) for x in wavelength_grids)
    wl_max = min(float(x[-1]) for x in wavelength_grids)
    if wl_max <= wl_min: raise ValueError("The selected AOIs have no overlapping wavelength range.")
    native = wavelength_grids[0]
    native = native[(native >= wl_min) & (native <= wl_max)]
    if len(native) < 2: raise ValueError("The reference AOI has fewer than two wavelengths in the shared range.")
    largest_gap_nm = float(np.max(np.diff(native)))
    if interpolation_step_nm is None:
        return native.copy(),False,largest_gap_nm

    target = [float(native[0])]
    n_inserted = 0
    for left,right in zip(native[:-1],native[1:]):
        gap = float(right-left)
        # ceil(gap/step)-1 is zero for gap <= step, one for step < gap <= 2*step, etc.
        n_local = max(0,int(np.ceil(gap/interpolation_step_nm-1e-12))-1)
        if n_local:
            inserted = left+interpolation_step_nm*np.arange(1,n_local+1,dtype=float)
            inserted = inserted[inserted < right-1e-12]
            target.extend(inserted.tolist())
            n_inserted += len(inserted)
        target.append(float(right))
    return np.asarray(target,dtype=float),bool(n_inserted),largest_gap_nm

def smooth_and_interpolate_exp(data,aoi_deg,target_wl_nm,smooth,interpolate=False): # smooth=False for no smoothing, or smooth=x for Gaussian sigma in nm
    """Prepare one AOI on the common grid while retaining its native points for plotting.
    When "interpolate" is true, Psi and unwrapped Delta use shape-preserving
    cubic interpolation. Measurement uncertainties remain linear to avoid
    negative or oscillatory sigma values.
    """
    df = data[np.isclose(data["AOI_(°)"],aoi_deg)].sort_values("lambda_(nm)").copy()
    # If two concatenated datasets contain the same wavelength, simply keep the first measurement.
    n_duplicates = df.duplicated(subset="lambda_(nm)",keep="first").sum()
    df = df.drop_duplicates(subset="lambda_(nm)",keep="first")
    x = df["lambda_(nm)"].to_numpy(float)
    psi_rad = np.deg2rad(df["psi_(°)"].to_numpy(float))
    delta_rad = np.deg2rad(df["delta_(°)"].to_numpy(float))
    psi_sigma_deg = df["psi_sigma_(°)"].to_numpy(float)
    delta_sigma_deg = df["delta_sigma_(°)"].to_numpy(float)
    def interpolate_missing(y,name):
        y = np.asarray(y,dtype=float)
        valid = np.isfinite(y)
        n_missing = np.sum(~valid)
        if n_missing == 0:
            return y
        if np.sum(valid) < 2:
            raise ValueError(f"AOI {aoi_deg:g}°: not enough valid {name} values to interpolate.")
        y = y.copy()
        y[~valid] = np.interp(x[~valid],x[valid],y[valid])
        return y
    # Psi is not periodic, so ordinary interpolation is sufficient.
    psi_rad = interpolate_missing(psi_rad,"Psi")
    # Delta is periodic. First unwrap only the actually measured finite values.
    # Then interpolate missing values on the continuous, unwrapped phase.
    valid_delta = np.isfinite(delta_rad)
    if np.sum(valid_delta) < 2:
        raise ValueError(f"AOI {aoi_deg:g}°: not enough valid Delta values to unwrap/interpolate.")
    delta_unwrapped_valid = np.unwrap(delta_rad[valid_delta])
    delta_rad_unwrapped = np.full_like(delta_rad,np.nan,dtype=float)
    delta_rad_unwrapped[valid_delta] = delta_unwrapped_valid
    delta_rad = interpolate_missing(delta_rad_unwrapped,"Delta")
    # Interpolate missing uncertainty values as well.
    psi_sigma_deg = interpolate_missing(psi_sigma_deg,"Psi sigma")
    delta_sigma_deg = interpolate_missing(delta_sigma_deg,"Delta sigma")

    target_wl_nm = np.asarray(target_wl_nm,dtype=float)
    if np.any(target_wl_nm < x[0]-1e-10) or np.any(target_wl_nm > x[-1]+1e-10):
        raise ValueError(f"AOI {aoi_deg:g}°: the target wavelength grid would require extrapolation.")

    def evaluate_signal(y):
        if interpolate:
            return PchipInterpolator(x,y,extrapolate=False)(target_wl_nm)
        return np.interp(target_wl_nm,x,y)

    result = {
        "psi_rad":evaluate_signal(psi_rad),
        "delta_rad":evaluate_signal(delta_rad),
        "psi_sigma_deg":np.interp(target_wl_nm,x,psi_sigma_deg),
        "delta_sigma_deg":np.interp(target_wl_nm,x,delta_sigma_deg),
        "raw_wl_nm":x.copy(),
        "psi_rad_raw":psi_rad.copy(),
        "delta_rad_raw":delta_rad.copy(),
        "psi_sigma_deg_raw":psi_sigma_deg.copy(),
        "delta_sigma_deg_raw":delta_sigma_deg.copy(),
        "target_wl_nm":target_wl_nm.copy(),
        "spline_interpolated":bool(interpolate),
        "smoothing_nm":False if not smooth else float(smooth),
    }
    result["rho"] = np.tan(result["psi_rad"])*np.exp(-1j*result["delta_rad"])
    if smooth:
        sigma_nm = float(smooth)
        dx_nm = np.median(np.diff(x))
        sigma_samples = max(sigma_nm/dx_nm,1e-12)
        psi_rad_smooth = gaussian_filter1d(psi_rad,sigma=sigma_samples,mode="nearest")
        delta_rad_smooth = gaussian_filter1d(delta_rad,sigma=sigma_samples,mode="nearest")
        psi_sigma_deg_smooth = gaussian_filter1d(psi_sigma_deg,sigma=sigma_samples,mode="nearest")
        delta_sigma_deg_smooth = gaussian_filter1d(delta_sigma_deg,sigma=sigma_samples,mode="nearest")
        result.update({
            "psi_rad_smooth":evaluate_signal(psi_rad_smooth),
            "delta_rad_smooth":evaluate_signal(delta_rad_smooth),
            "psi_sigma_deg_smooth":np.interp(target_wl_nm,x,psi_sigma_deg_smooth),
            "delta_sigma_deg_smooth":np.interp(target_wl_nm,x,delta_sigma_deg_smooth),
        })
    return result

def interpolate_dielectric_function(df,target_wl_nm,material_name="material"):
    """Linearly interpolate isotropic n,k or a uniaxial "o/e" data pair."""
    if isinstance(df,dict):
        aliases = (("o","e"),("ordinary","extraordinary"),("parallel","perpendicular"))
        pair = next(((left,right) for left,right in aliases if left in df and right in df),None)
        if pair is None:
            raise ValueError(f"{material_name}: anisotropic optical data must contain an o/e, ordinary/extraordinary, or parallel/perpendicular pair.")
        left,right = pair
        return {
            "o":interpolate_dielectric_function(df[left],target_wl_nm,material_name=f"{material_name}_o"),
            "e":interpolate_dielectric_function(df[right],target_wl_nm,material_name=f"{material_name}_e"),
        }
    if not isinstance(df,pd.DataFrame): raise TypeError(f"{material_name}: optical data must be supplied as a pandas DataFrame.")
    required = ["wl","n"]
    missing = [column for column in required if column not in df.columns]
    if missing: raise ValueError(f"{material_name}: optical table is missing required column(s): {missing}")
    w = df["wl"].to_numpy(dtype=float)*1e3
    n = df["n"].to_numpy(dtype=float)
    k = df["k"].to_numpy(dtype=float) if "k" in df.columns else np.zeros_like(n,dtype=float)
    valid = np.isfinite(w) & np.isfinite(n) & np.isfinite(k)
    if np.count_nonzero(valid) < 2: raise ValueError(f"{material_name}: fewer than two finite n,k rows are available.")
    w,n,k = w[valid],n[valid],k[valid]
    order = np.argsort(w)
    w,n,k = w[order],n[order],k[order]
    # If a table itself contains duplicate wavelengths, keep the first tabulated value.
    unique_w,unique_idx = np.unique(w,return_index=True)
    w,n,k = unique_w,n[unique_idx],k[unique_idx]
    target_wl_nm = np.asarray(target_wl_nm,dtype=float)
    data_min,data_max = float(w[0]),float(w[-1])
    exp_min,exp_max = float(np.min(target_wl_nm)),float(np.max(target_wl_nm))
    if exp_min < data_min or exp_max > data_max:
        raise ValueError(f"{material_name}: experimental range {exp_min:.2f}–{exp_max:.2f} nm "f"extends outside optical-data range {data_min:.2f}–{data_max:.2f} nm. " "Extrapolation is disabled.")
    n_interp = interp1d(w,n,kind="linear",bounds_error=True,assume_sorted=True)(target_wl_nm)
    k_interp = interp1d(w,k,kind="linear",bounds_error=True,assume_sorted=True)(target_wl_nm)
    N = n_interp+1j*k_interp
    return N**2

def measurement_bandwidth(data, wl_nm):
    """Return the local spectral bandwidth in nm, falling back to the wavelength sampling step."""
    wl_nm = np.asarray(wl_nm,dtype=float)
    wl_step_nm = np.median(np.diff(np.sort(wl_nm))) if len(wl_nm) > 1 else 1.0
    if "bandwidth_(nm)" in data.columns:
        bandwidth_table = data.groupby("lambda_(nm)")["bandwidth_(nm)"].median().sort_index()
        bandwidth_nm = np.interp(wl_nm,bandwidth_table.index.to_numpy(float),bandwidth_table.to_numpy(float))
    else:
        bandwidth_nm = np.full_like(wl_nm,wl_step_nm,dtype=float)
    return np.where(np.isfinite(bandwidth_nm) & (bandwidth_nm > 0),bandwidth_nm,wl_step_nm)

def spectral_quadrature_weights(wl_nm,mode="points",hc_ev_nm=1239.8419843320026):
    """Return mean-one spectral weights for point, wavelength, or energy weighting."""
    wl_nm = np.asarray(wl_nm,dtype=float)
    mode = str(mode).strip().lower()
    if mode == "points":
        return np.ones_like(wl_nm,dtype=float)
    if mode not in ("wavelength","energy"):
        raise ValueError('spectral weighting must be "points", "wavelength", or "energy".')
    if wl_nm.size < 2:
        return np.ones_like(wl_nm,dtype=float)
    coordinate = wl_nm if mode == "wavelength" else float(hc_ev_nm)/wl_nm
    spacing = np.abs(np.diff(coordinate))
    weights = np.empty_like(coordinate,dtype=float)
    weights[0] = 0.5*spacing[0]
    weights[-1] = 0.5*spacing[-1]
    if wl_nm.size > 2:
        weights[1:-1] = 0.5*(spacing[:-1]+spacing[1:])
    mean_weight = float(np.mean(weights))
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Could not construct finite spectral weights from the wavelength grid.")
    return weights/mean_weight

def build_measurement_data(meas,AOIs,wl_nm,spectral_weighting="points",hc_ev_nm=1239.8419843320026):
    """Collect the fitted Psi/Delta arrays and their spectral objective weights."""
    AOIs = tuple(AOIs)
    wl_nm = np.asarray(wl_nm,dtype=float)
    use_smooth = "psi_rad_smooth" in meas[AOIs[0]] and "delta_rad_smooth" in meas[AOIs[0]]
    psi_key = "psi_rad_smooth" if use_smooth else "psi_rad"
    delta_key = "delta_rad_smooth" if use_smooth else "delta_rad"
    psi_sigma_key = "psi_sigma_deg_smooth" if use_smooth else "psi_sigma_deg"
    delta_sigma_key = "delta_sigma_deg_smooth" if use_smooth else "delta_sigma_deg"
    mode = str(spectral_weighting).strip().lower()
    return MeasurementData(
        wl_nm=wl_nm,
        AOIs=AOIs,
        psi_exp_deg={aoi:np.rad2deg(meas[aoi][psi_key]) for aoi in AOIs},
        delta_exp_deg={aoi:np.mod(np.rad2deg(meas[aoi][delta_key]),360.0) for aoi in AOIs},
        psi_sigma_deg={aoi:np.asarray(meas[aoi][psi_sigma_key],dtype=float) for aoi in AOIs},
        delta_sigma_deg={aoi:np.asarray(meas[aoi][delta_sigma_key],dtype=float) for aoi in AOIs},
        spectral_weighting=mode,
        spectral_weights=spectral_quadrature_weights(wl_nm,mode=mode,hc_ev_nm=hc_ev_nm),
    )

