# -*- coding: utf-8 -*-
"""Random-forest WUEt extrapolation module for RTEM.

Main features
-------------
1. Static predictors with dimensions (lat, lon) are broadcast to
   (lat, lon, timestamp) only when needed.
2. The Baseflag and CSWI-based training-sample selection logic is retained.
3. The module outputs RTEM_WUE, RTEM_T, RTEM_E, feature importances, and
   out-of-bag scores.
4. Debug outputs are optional and disabled by default.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor


DEFAULT_RF_FEATURES = [
    "Qv", "temp", "shum", "wind", "VPD",
    "GPPgradtime", "Rg_pot", "Rgpot_grady", "Rgpot_gradtime",
    "CSWI", "LAI",
    "SWC", "pres", "Rn",
    "pH", "SOC", "CEC",
    "clay", "silt", "sand",
    "bd", "cf", "thickness",
    "tn", "tp", "tk"
]


DEFAULT_RF_KWARGS = {
    "n_estimators": 300,
    "oob_score": True,
    "max_features": "sqrt",
    "min_samples_leaf": 3,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 0,
}


def _get_feature_3d(ds: xr.Dataset, var_name: str) -> np.ndarray:
    """Return a feature as a 3D array with dimensions (lat, lon, timestamp)."""
    if var_name not in ds:
        raise KeyError(f"Feature variable not found in dataset: {var_name}")

    arr = ds[var_name].values
    n_time = ds.sizes["timestamp"]

    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, np.newaxis], n_time, axis=2).astype(np.float32)
    raise ValueError(f"Unsupported dimensions for variable {var_name}: {arr.shape}")


def build_feature_cube(ds: xr.Dataset, feature_names: Sequence[str]) -> np.ndarray:
    feature_list = []
    for name in feature_names:
        feature_list.append(_get_feature_3d(ds, name))
    feature_cube = np.stack(feature_list, axis=-1)  # (lat, lon, timestamp, n_features)
    feature_cube[feature_cube <= -9000] = np.nan
    return feature_cube


def create_nanflag(ds: xr.Dataset, feature_names: Sequence[str]) -> np.ndarray:
    nanflag = np.isfinite(ds["ET"].values) & np.isfinite(ds["GPP"].values) & np.isfinite(ds["WUEt"].values)
    for name in feature_names:
        arr = _get_feature_3d(ds, name)
        nanflag &= np.isfinite(arr)
        nanflag &= (arr > -9000)
    return nanflag.astype(bool)


def fit_predict_wuet(
    ds: xr.Dataset,
    feature_names: Optional[Sequence[str]] = None,
    cswi_limits: Optional[Sequence[float]] = None,
    rf_kwargs: Optional[Dict] = None,
    min_train_samples: int = 500,
    rf_flag_output_dir: Optional[str] = None,
) -> xr.Dataset:
    feature_names = list(feature_names) if feature_names is not None else list(DEFAULT_RF_FEATURES)
    cswi_limits = np.array(cswi_limits if cswi_limits is not None else [-0.5], dtype=np.float32)

    for name in feature_names:
        if name not in ds:
            raise KeyError(f"Missing feature: {name}")

    kwargs = dict(DEFAULT_RF_KWARGS)
    if rf_kwargs is not None:
        kwargs.update(rf_kwargs)

    feature_cube = build_feature_cube(ds, feature_names)
    nanflag = create_nanflag(ds, feature_names)

    if "qualityFlag" not in ds:
        raise KeyError("qualityFlag is missing. Run preprocess_dataset first.")
    if "vegFlag" not in ds or "NDVIFlag" not in ds or "posFlag" not in ds:
        raise KeyError("Required screening flags are missing. Run preprocess_dataset first.")

    baseflag = (
        ds["qualityFlag"].values
        & ds["vegFlag"].values
        & ds["NDVIFlag"].values
        & ds["posFlag"].values
        & nanflag
    )

    out = ds.copy()
    out.coords["CSWIlims"] = cswi_limits
    out.coords["RFmod_vars"] = np.array(feature_names, dtype=object)

    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    n_time = ds.sizes["timestamp"]
    n_limit = len(cswi_limits)
    n_feat = len(feature_names)

    out["nanflag"] = (("lat", "lon", "timestamp"), nanflag)
    out["Baseflag"] = (("lat", "lon", "timestamp"), baseflag)
    out["RFflag"] = (("lat", "lon", "timestamp", "CSWIlims"), np.zeros((n_lat, n_lon, n_time, n_limit), dtype=bool))

    fill = np.full((n_lat, n_lon, n_time, n_limit), np.nan, dtype=np.float32)
    out["RTEM_WUE"] = (("lat", "lon", "timestamp", "CSWIlims"), fill.copy())
    out["RTEM_T"] = (("lat", "lon", "timestamp", "CSWIlims"), fill.copy())
    out["RTEM_E"] = (("lat", "lon", "timestamp", "CSWIlims"), fill.copy())

    out["feature_importances"] = (("RFmod_vars", "CSWIlims"), np.full((n_feat, n_limit), np.nan, dtype=np.float32))
    out["NumForestPoints"] = (("CSWIlims",), np.full((n_limit,), np.nan, dtype=np.float32))
    out["oob_scores"] = (("CSWIlims",), np.full((n_limit,), np.nan, dtype=np.float32))

    y_all = ds["WUEt"].values.astype(np.float32)

    for idx, limit in enumerate(cswi_limits):
        cur_flag = baseflag & (ds["CSWI"].values < limit) & ds["WUEtFlag"].values
        out["RFflag"].values[:, :, :, idx] = cur_flag
        out["NumForestPoints"].values[idx] = float(cur_flag.sum())


        if rf_flag_output_dir is not None:
            flag_dir = Path(rf_flag_output_dir)
            flag_dir.mkdir(parents=True, exist_ok=True)
            np.save(flag_dir / f"cur_flag_CSWI_{limit}.npy", cur_flag)
            print(f"CSWI<{limit}: RF training mask saved to {flag_dir}")

        if cur_flag.sum() < min_train_samples:
            print(f"CSWI<{limit}: insufficient training samples; skipped. Current n={cur_flag.sum()}")
            continue

        train_x = feature_cube[cur_flag]
        train_y = y_all[cur_flag]

        valid_y = np.isfinite(train_y) & (train_y > 0)
        train_x = train_x[valid_y]
        train_y = train_y[valid_y]

        if train_y.size < min_train_samples:
            print(f"CSWI<{limit}: insufficient samples after removing invalid targets; skipped. Current n={train_y.size}")
            continue

        forest = RandomForestRegressor(**kwargs)
        forest.fit(train_x, train_y)
        print(f"CSWI<{limit}: random forest training completed. Training n={train_y.size}")

        if hasattr(forest, "oob_score_"):
            out["oob_scores"].values[idx] = np.float32(forest.oob_score_)

        out["feature_importances"].values[:, idx] = forest.feature_importances_.astype(np.float32)

        pred_mask = nanflag
        pred_x = feature_cube[pred_mask]
        pred_wue = forest.predict(pred_x).astype(np.float32)

        pred_grid = np.full((n_lat, n_lon, n_time), np.nan, dtype=np.float32)
        pred_grid[pred_mask] = pred_wue
        out["RTEM_WUE"].values[:, :, :, idx] = pred_grid

        # T = GPP / WUEt; E = ET - T
        t_grid = np.full_like(pred_grid, np.nan)
        valid = np.isfinite(pred_grid) & (pred_grid > 0) & np.isfinite(ds["GPP"].values)
        t_grid[valid] = ds["GPP"].values[valid] / pred_grid[valid]
        e_grid = ds["ET"].values.astype(np.float32) - t_grid

        out["RTEM_T"].values[:, :, :, idx] = t_grid.astype(np.float32)
        out["RTEM_E"].values[:, :, :, idx] = e_grid.astype(np.float32)

    out.attrs["features"] = ",".join(feature_names)
    out.attrs["PartRunTimestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.attrs["PartType"] = os.path.basename(__file__)
    out.attrs["ModelType"] = "RandomForestRegressor"
    return out


if __name__ == "__main__":
    print("This is a modeling module. Import and call fit_predict_wuet from run_rtem.py.")
