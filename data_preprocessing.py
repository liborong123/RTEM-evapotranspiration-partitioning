# -*- coding: utf-8 -*-
"""Data preprocessing utilities for the RTEM workflow.

This module loads dynamic meteorological/vegetation variables, static soil
properties, and vegetation masks for regional WUEt and transpiration estimation.

Design principles
-----------------
1. Dynamic variables are stored as (lat, lon, timestamp).
2. Static soil variables are stored as (lat, lon).
3. Static variables are broadcast only when needed by the modeling module.
4. Derived variables are retained for consistency with the RTEM workflow,
   including CSWI, GPP gradients, and potential-radiation gradients.
5. Paths, variable names, and thresholds are centralized in ``DataConfig``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from osgeo import gdal, osr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


gdal.UseExceptions()

@dataclass
class DataConfig:
    meteorology_root: str = r"G:\caculate_T"
    soil_root: str = r"G:\soil_property\temp_origin"
    vegetation_root: str = r"G:\veg"
    start_date: str = "2015-01-01"
    end_date: str = "2020-12-31"
    precip_threshold_mm_day: float = 5.0
    ndvi90_threshold: float = 0.55
    vegetation_classes: Tuple[int, ...] = (10,)  # Default: grassland
    soil_var_patterns: Dict[str, Sequence[str]] = field(default_factory=lambda: {
        "pH": ("ph", "pH"),
        "SOC": ("soc",),
        "AN": ("an", "alkaline", "available_n"),
        "AP": ("ap", "available_p"),
    })
    dynamic_var_dirs: Dict[str, str] = field(default_factory=lambda: {
        "precip": "prec",
        "ET": "ET",
        "GPP": "GPP_PML",
        "VPD": "VPD",
        "LAI": "LAI",
        "wind": "wind",
        "Rn": "Rn",
        "temp": "temp",
        "shum": "shum",
        "WUEt": "uWUEt",
        "SWC": "SWC",
        "pres": "pres",
        "srad": "srad",
        "NDVI": "NDVI",
    })


# ==============================
# Basic utilities
# ==============================

def _ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


def _extract_date_from_filename(filepath: str) -> Optional[str]:
    base = os.path.basename(filepath)
    date_part = os.path.splitext(base)[0].split("_")[-1]
    try:
        datetime.strptime(date_part, "%Y-%m-%d")
        return date_part
    except ValueError:
        return None


def _list_tifs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.tif")))


def _read_band(path: str, nodata_to_nan: bool = True) -> np.ndarray:
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {path}")
    arr = ds.ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata_to_nan:
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr <= -9000] = np.nan
    return arr


def _get_lon_lat_grids(reference_tif: str) -> Tuple[np.ndarray, np.ndarray]:
    ds = gdal.Open(reference_tif)
    if ds is None:
        raise RuntimeError(f"Could not open reference raster: {reference_tif}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    transform = ds.GetGeoTransform()

    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(ds.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    ct = osr.CoordinateTransformation(prosrs, geosrs)

    lon = np.zeros((height, width), dtype=np.float32)
    lat = np.zeros((height, width), dtype=np.float32)

    for row in range(height):
        for col in range(width):
            x = transform[0] + col * transform[1] + row * transform[2]
            y = transform[3] + col * transform[4] + row * transform[5]
            lon_val, lat_val, _ = ct.TransformPoint(x, y)
            lon[row, col] = lon_val
            lat[row, col] = lat_val
    return lon, lat


# ==============================
# Potential radiation calculation
# ==============================

def timeconv(val: str) -> int:
    dt = datetime.strptime(val, "%Y-%m-%d")
    return dt.timetuple().tm_yday


def solar_declination(doy: int) -> np.ndarray:
    frac_year_rad = 2 * np.pi * (doy - 1) / 365.24
    sol_decl_rad = (
        (0.33281 - 22.984 * np.cos(frac_year_rad) - 0.34990 * np.cos(2 * frac_year_rad)
         - 0.13980 * np.cos(3 * frac_year_rad) + 3.7872 * np.sin(frac_year_rad)
         + 0.03205 * np.sin(2 * frac_year_rad) + 0.07187 * np.sin(3 * frac_year_rad))
        / 180 * np.pi
    )
    return sol_decl_rad


def day_time(local_lat: np.ndarray, sol_decl_rad: np.ndarray) -> np.ndarray:
    term = np.tan(-sol_decl_rad) * np.tan(local_lat / 180 * np.pi)
    term = np.clip(term, -1.0, 1.0)
    return np.arccos(term) * 2


def solar_altitude(local_lat: np.ndarray, sol_decl_rad: np.ndarray, daytime: np.ndarray) -> np.ndarray:
    mean_solar_hour_angle = 2 * np.sin(daytime / 2) / np.maximum(daytime, 1e-6)
    return np.arcsin(
        np.sin(sol_decl_rad) * np.sin(local_lat / 180 * np.pi)
        + np.cos(sol_decl_rad) * np.cos(local_lat / 180 * np.pi) * np.cos(mean_solar_hour_angle)
    )


def calc_ext_radiation(doy: int) -> np.ndarray:
    frac_year_rad = 2 * np.pi * (doy - 1) / 365.24
    solar_irr_wm2 = 1366.1
    return solar_irr_wm2 * (
        1.00011 + 0.034221 * np.cos(frac_year_rad) + 0.00128 * np.sin(frac_year_rad)
        + 0.000719 * np.cos(2 * frac_year_rad) + 0.000077 * np.sin(2 * frac_year_rad)
    )


def calc_potential_radiation(date_str: str, local_lat: np.ndarray) -> np.ndarray:
    doy = timeconv(date_str)
    sol_decl_rad = solar_declination(doy)
    daytime = day_time(local_lat, sol_decl_rad)
    sol_elev_rad = solar_altitude(local_lat, sol_decl_rad, daytime)
    ext_radiation = calc_ext_radiation(doy)
    pot_radiation = ext_radiation * np.sin(sol_elev_rad)
    return np.maximum(pot_radiation, 0.0).astype(np.float32)


# ==============================
# Dynamic and static data loading
# ==============================

def _collect_available_dates(folder: str) -> Dict[str, str]:
    files = _list_tifs(folder)
    date_to_file: Dict[str, str] = {}
    for file in files:
        date_str = _extract_date_from_filename(file)
        if date_str is not None:
            date_to_file[date_str] = file
    return date_to_file


def collect_common_dates(config: DataConfig) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    date_maps: Dict[str, Dict[str, str]] = {}
    common_dates: Optional[set] = None

    for var_name, subdir in config.dynamic_var_dirs.items():
        folder = os.path.join(config.meteorology_root, subdir)
        _ensure_exists(folder)
        mapping = _collect_available_dates(folder)
        mapping = {
            d: f for d, f in mapping.items()
            if config.start_date <= d <= config.end_date
        }
        if not mapping:
            raise RuntimeError(f"No dated files were found for variable {var_name} in {folder}")
        date_maps[var_name] = mapping
        common_dates = set(mapping.keys()) if common_dates is None else (common_dates & set(mapping.keys()))

    if common_dates is None:
        raise RuntimeError("No dates were found for dynamic variables")

    common_dates = sorted(common_dates)
    if not common_dates:
        raise RuntimeError("Dynamic variables have no common dates. Check missing files.")
    return common_dates, date_maps


def load_dynamic_data(config: DataConfig, common_dates: List[str], date_maps: Dict[str, Dict[str, str]]) -> Dict[str, np.ndarray]:
    data: Dict[str, List[np.ndarray]] = {k: [] for k in config.dynamic_var_dirs.keys()}

    for date_str in tqdm(common_dates, desc="Loading dynamic variables"):
        for var_name in config.dynamic_var_dirs.keys():
            arr = _read_band(date_maps[var_name][date_str])
            data[var_name].append(arr[:, :, np.newaxis])

    stacked = {k: np.concatenate(v, axis=2).astype(np.float32) for k, v in data.items()}

    # Unit conversions retained from the original workflow
    stacked["precip"] = stacked["precip"] * 24.0
    stacked["precip"][stacked["precip"] < config.precip_threshold_mm_day] = 0.0

    stacked["temp"] = stacked["temp"] - 273.15
    stacked["Rn"] = stacked["Rn"] * 0.01
    stacked["SWC"] = stacked["SWC"] * 0.0001

    # Derived vegetation-available radiation term
    stacked["Qv"] = stacked["Rn"] * (1.0 - np.exp(-0.65 * stacked["LAI"]))
    return stacked


def _find_first_matching_tif(root: str, patterns: Sequence[str]) -> str:
    tif_files = _list_tifs(root)
    if not tif_files:
        raise RuntimeError(f"No tif files were found in soil directory: {root}")

    lowered_patterns = [p.lower() for p in patterns]
    for tif in tif_files:
        name = os.path.basename(tif).lower()
        if any(p in name for p in lowered_patterns):
            return tif
    raise RuntimeError(f"No soil file matching {patterns} was found in {root}")


def load_static_soil_data(config: DataConfig) -> Dict[str, np.ndarray]:
    _ensure_exists(config.soil_root)
    soil_data: Dict[str, np.ndarray] = {}
    for soil_name, patterns in config.soil_var_patterns.items():
        tif = _find_first_matching_tif(config.soil_root, patterns)
        soil_data[soil_name] = _read_band(tif)
    return soil_data


def load_vegetation_mask(config: DataConfig, n_time: int) -> np.ndarray:
    _ensure_exists(config.vegetation_root)
    tif_files = _list_tifs(config.vegetation_root)
    if not tif_files:
        raise RuntimeError(f"No vegetation tif files were found in: {config.vegetation_root}")

    selected = []
    for tif in tif_files:
        base = os.path.basename(tif)
        year_candidates = [token[:4] for token in base.split("_") if len(token) >= 4 and token[:4].isdigit()]
        if year_candidates:
            year = int(year_candidates[-1])
            if 2016 <= year <= 2020:
                selected.append(tif)

    if not selected:
        selected = tif_files

    masks = []
    for tif in selected:
        arr = _read_band(tif, nodata_to_nan=False)
        mask = np.isin(arr.astype(np.int32), np.array(config.vegetation_classes, dtype=np.int32))
        masks.append(mask)

    veg_flag_2d = np.logical_and.reduce(masks).astype(bool)
    veg_flag_3d = np.repeat(veg_flag_2d[:, :, np.newaxis], n_time, axis=2)
    return veg_flag_3d


# ==============================
# Feature engineering and quality flags
# ==============================

def mad_based_valid_flag(data: np.ndarray, thresh: float = 3.0, group_size: int = 15) -> np.ndarray:
    """Return a validity mask; True indicates retained values and False indicates outliers."""
    valid = np.ones_like(data, dtype=bool)
    h, w, t = data.shape
    for i in tqdm(range(h), desc="MAD outlier filtering"):
        for j in range(w):
            series = data[i, j, :]
            if np.all(~np.isfinite(series)):
                valid[i, j, :] = False
                continue
            flags = []
            for start in range(0, t, group_size):
                seg = series[start:start + group_size]
                good = np.isfinite(seg)
                if not np.any(good):
                    flags.extend([False] * len(seg))
                    continue
                med = np.nanmedian(seg)
                dev = np.abs(seg - med)
                mad = np.nanmedian(dev) * 1.4826
                if not np.isfinite(mad) or mad == 0:
                    seg_flag = np.isfinite(seg)
                else:
                    threshold = thresh * mad
                    seg_flag = np.isfinite(seg) & (seg >= med - threshold) & (seg <= med + threshold)
                flags.extend(seg_flag.tolist())
            valid[i, j, :] = np.array(flags[:t], dtype=bool)
    return valid


def calc_cswi(precip: np.ndarray, et: np.ndarray, s0: float = 5.0) -> np.ndarray:
    if precip.shape != et.shape:
        raise ValueError("precip and ET must have the same shape")
    cswi = np.zeros_like(precip, dtype=np.float32)
    cswi[:, :, 0] = s0
    h, w, t = et.shape
    for i in tqdm(range(h), desc="Calculating CSWI"):
        for j in range(w):
            for k in range(t - 1):
                l = k + 1
                step_val = min(cswi[i, j, k] + precip[i, j, l] - et[i, j, l], s0)
                if precip[i, j, l] > 0:
                    cswi[i, j, l] = max(step_val, min(precip[i, j, l], s0))
                else:
                    cswi[i, j, l] = step_val
    return cswi


def calc_gpp_gradient(gpp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = gpp.copy()
    arr[~np.isfinite(arr)] = 0
    arr[arr < -9000] = 0
    grad_y, grad_x, grad_t = np.gradient(gaussian_filter(arr, sigma=[0, 0, 0.12]))
    return grad_y.astype(np.float32), grad_x.astype(np.float32), grad_t.astype(np.float32)


def calc_ndvi90_flag(ndvi: np.ndarray, threshold: float) -> np.ndarray:
    p90 = np.nanpercentile(ndvi, 90, axis=2)
    flag2d = p90 > threshold
    flag3d = np.repeat(flag2d[:, :, np.newaxis], ndvi.shape[2], axis=2)
    return flag3d.astype(bool)


def build_dataset(
    timestamp: Sequence[str],
    dynamic_data: Dict[str, np.ndarray],
    lon_2d: np.ndarray,
    lat_2d: np.ndarray,
    soil_data: Optional[Dict[str, np.ndarray]] = None,
    veg_flag_3d: Optional[np.ndarray] = None,
) -> xr.Dataset:
    """Build a unified ``xarray.Dataset``.

    Dynamic variables use dimensions (lat, lon, timestamp).
    Static variables use dimensions (lat, lon).
    """
    coords = {
        "lat": lat_2d[:, 0],
        "lon": lon_2d[0, :],
        "timestamp": np.array(timestamp, dtype="datetime64[ns]"),
    }

    data_vars = {}
    for var_name, arr in dynamic_data.items():
        if arr.ndim != 3:
            raise ValueError(f"Dynamic variable {var_name} must be 3D; got shape {arr.shape}")
        data_vars[var_name] = (("lat", "lon", "timestamp"), arr)

    data_vars["longitude"] = (("lat", "lon"), lon_2d.astype(np.float32))
    data_vars["latitude"] = (("lat", "lon"), lat_2d.astype(np.float32))

    if soil_data is not None:
        for var_name, arr in soil_data.items():
            if arr.ndim != 2:
                raise ValueError(f"Static variable {var_name} must be 2D; got shape {arr.shape}")
            data_vars[var_name] = (("lat", "lon"), arr.astype(np.float32))

    if veg_flag_3d is not None:
        data_vars["vegFlag"] = (("lat", "lon", "timestamp"), veg_flag_3d.astype(bool))

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["created_by"] = "data_preprocessing_v2.py"
    ds.attrs["note"] = "Dynamic variables are 3D, static soil variables are 2D, and static variables are broadcast in the modeling module."
    return ds


def preprocess_dataset(ds: xr.Dataset, ndvi90_threshold: float = 0.55) -> xr.Dataset:
    n_time = ds.sizes["timestamp"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]

    # Core derived variables
    ds["CSWI"] = (("lat", "lon", "timestamp"), calc_cswi(ds["precip"].values, ds["ET"].values))

    gpp_grady, gpp_gradx, gpp_gradtime = calc_gpp_gradient(ds["GPP"].values)
    ds["GPPgrady"] = (("lat", "lon", "timestamp"), gpp_grady)
    ds["GPPgradx"] = (("lat", "lon", "timestamp"), gpp_gradx)
    ds["GPPgradtime"] = (("lat", "lon", "timestamp"), gpp_gradtime)

    rgpot_grady = np.gradient(ds["Rg_pot"].values)[0].astype(np.float32)
    rgpot_gradtime = np.gradient(ds["Rg_pot"].values)[2].astype(np.float32)
    ds["Rgpot_grady"] = (("lat", "lon", "timestamp"), rgpot_grady)
    ds["Rgpot_gradtime"] = (("lat", "lon", "timestamp"), rgpot_gradtime)

    # Time variables
    ts = ds["timestamp"].values.astype("datetime64[D]")
    years = np.array([int(str(t)[:4]) for t in ts], dtype=np.int16)
    doys = np.array([datetime.strptime(str(t), "%Y-%m-%d").timetuple().tm_yday for t in ts], dtype=np.int16)
    sin_doy = np.sin(2 * np.pi * doys / 365.25).astype(np.float32)
    cos_doy = np.cos(2 * np.pi * doys / 365.25).astype(np.float32)

    ds["year"] = (("lat", "lon", "timestamp"), np.broadcast_to(years[np.newaxis, np.newaxis, :], (n_lat, n_lon, n_time)).copy())
    ds["DOY"] = (("lat", "lon", "timestamp"), np.broadcast_to(doys[np.newaxis, np.newaxis, :], (n_lat, n_lon, n_time)).copy())
    ds["sinDOY"] = (("lat", "lon", "timestamp"), np.broadcast_to(sin_doy[np.newaxis, np.newaxis, :], (n_lat, n_lon, n_time)).copy())
    ds["cosDOY"] = (("lat", "lon", "timestamp"), np.broadcast_to(cos_doy[np.newaxis, np.newaxis, :], (n_lat, n_lon, n_time)).copy())

    # Quality and screening flags
    ds["posFlag"] = (("lat", "lon", "timestamp"), ((ds["GPP"].values > 0) & (ds["ET"].values > 0)).astype(bool))
    ds["ETFlag"] = (("lat", "lon", "timestamp"), (ds["ET"].values > 0).astype(bool))
    ds["GPPFlag"] = (("lat", "lon", "timestamp"), (ds["GPP"].values > 0).astype(bool))
    ds["NDVIFlag"] = (("lat", "lon", "timestamp"), calc_ndvi90_flag(ds["NDVI"].values, ndvi90_threshold))
    ds["WUEtFlag"] = (("lat", "lon", "timestamp"), (mad_based_valid_flag(ds["WUEt"].values) & (ds["WUEt"].values < 100)).astype(bool))

    if "vegFlag" not in ds:
        ds["vegFlag"] = (("lat", "lon", "timestamp"), np.ones((n_lat, n_lon, n_time), dtype=bool))

    quality_flag = np.ones((n_lat, n_lon, n_time), dtype=bool)
    for var in ["ET", "GPP", "temp", "VPD", "LAI", "WUEt", "Qv", "precip", "shum", "wind", "Rn", "SWC", "pres", "Rg_pot", "NDVI"]:
        if var in ds:
            arr = ds[var].values
            quality_flag &= np.isfinite(arr)
            quality_flag &= (arr > -9000)
    ds["qualityFlag"] = (("lat", "lon", "timestamp"), quality_flag)

    ds.attrs["preprocessed_by"] = "preprocess_dataset"
    ds.attrs["ndvi90_threshold"] = float(ndvi90_threshold)
    return ds


# ==============================
# One-step data loading
# ==============================

def load_all_data(config: Optional[DataConfig] = None) -> xr.Dataset:
    config = config or DataConfig()

    common_dates, date_maps = collect_common_dates(config)
    dynamic_data = load_dynamic_data(config, common_dates, date_maps)

    # Use the first available raster from any dynamic variable as the coordinate reference
    ref_var = next(iter(config.dynamic_var_dirs.keys()))
    ref_tif = date_maps[ref_var][common_dates[0]]
    lon_2d, lat_2d = _get_lon_lat_grids(ref_tif)

    # Potential radiation
    rg_pot_list = []
    for date_str in tqdm(common_dates, desc="Calculating potential radiation Rg_pot"):
        rg_pot_list.append(calc_potential_radiation(date_str, lat_2d)[:, :, np.newaxis])
    dynamic_data["Rg_pot"] = np.concatenate(rg_pot_list, axis=2).astype(np.float32)

    soil_data = load_static_soil_data(config)
    veg_flag_3d = load_vegetation_mask(config, n_time=len(common_dates))

    ds = build_dataset(
        timestamp=common_dates,
        dynamic_data=dynamic_data,
        lon_2d=lon_2d,
        lat_2d=lat_2d,
        soil_data=soil_data,
        veg_flag_3d=veg_flag_3d,
    )
    ds = preprocess_dataset(ds, ndvi90_threshold=config.ndvi90_threshold)
    return ds


if __name__ == "__main__":
    cfg = DataConfig()
    ds = load_all_data(cfg)
    print(ds)
    print("Data preprocessing completed.")
