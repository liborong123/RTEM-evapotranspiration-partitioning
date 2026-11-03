"""Random forest estimation of regional transpiration."""

from __future__ import annotations
import datetime
import os
from typing import Dict, List, Optional, Sequence
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

DEFAULT_RF_FEATURES = [
    "temp",
    "shum",
    "wind",
    "GPPgradtime",
    "Rg_pot",
    "Rgpot_grady",
    "Rgpot_gradtime",
    "CSWI",
    "LAI",
    "pres",
    "Rn",
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
    """Return an array with dimensions (lat, lon, timestamp)."""
    if var_name not in ds:
        raise KeyError(f"Feature not found: {var_name}")
    arr = ds[var_name].values
    n_time = ds.sizes["timestamp"]
    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, np.newaxis], n_time, axis=2).astype(np.float32)
    raise ValueError(f"Unsupported shape for {var_name}: {arr.shape}")


def build_feature_cube(ds: xr.Dataset, feature_names: Sequence[str]) -> np.ndarray:
    feature_list = []
    for name in feature_names:
        feature_list.append(_get_feature_3d(ds, name))
    feature_cube = np.stack(feature_list, axis=-1)
    feature_cube[feature_cube <= -9000] = np.nan
    return feature_cube


def create_nanflag(ds: xr.Dataset, feature_names: Sequence[str]) -> np.ndarray:
    nanflag = (
        np.isfinite(ds["ET"].values)
        & np.isfinite(ds["GPP"].values)
        & np.isfinite(ds["WUEt"].values)
    )
    for name in feature_names:
        arr = _get_feature_3d(ds, name)
        nanflag &= np.isfinite(arr)
        nanflag &= arr > -9000
    return nanflag.astype(bool)


def predict_forest_percentile(
    forest: RandomForestRegressor,
    features: np.ndarray,
    percentile: float,
    chunk_size: int = 100000,
) -> np.ndarray:
    """Calculate a pointwise percentile of the individual tree predictions."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(
            f"prediction_percentile must be between 0 and 100; received {percentile}"
        )
    if chunk_size < 1:
        raise ValueError("prediction_chunk_size must be a positive integer")
    result = np.empty(features.shape[0], dtype=np.float32)
    n_trees = len(forest.estimators_)
    n_jobs = forest.n_jobs if forest.n_jobs is not None else 1
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        chunk = features[start:end]
        tree_predictions = np.empty((n_trees, end - start), dtype=np.float32)

        def predict_one(tree_index: int) -> None:
            tree_predictions[tree_index] = (
                forest.estimators_[tree_index]
                .predict(chunk)
                .astype(np.float32, copy=False)
            )

        Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(
            (delayed(predict_one)(tree_index) for tree_index in range(n_trees))
        )
        result[start:end] = np.percentile(tree_predictions, percentile, axis=0).astype(
            np.float32, copy=False
        )
        chunk_number = start // chunk_size + 1
        chunk_total = (features.shape[0] + chunk_size - 1) // chunk_size
        if chunk_number % 10 == 0 or end == features.shape[0]:
            print(
                f"P{percentile:g} prediction: {end}/{features.shape[0]} (chunk {chunk_number}/{chunk_total})",
                flush=True,
            )
    return result


def fit_predict_wuet(
    ds: xr.Dataset,
    feature_names: Optional[Sequence[str]] = None,
    cswi_limits: Optional[Sequence[float]] = None,
    rf_kwargs: Optional[Dict] = None,
    min_train_samples: int = 500,
    curflag_dir: Optional[str] = None,
    prediction_percentile: Optional[float] = 85.0,
    prediction_chunk_size: int = 100000,
    model_path: Optional[str] = None,
    baseline_training_mask: Optional[np.ndarray] = None,
) -> xr.Dataset:
    feature_names = (
        list(feature_names) if feature_names is not None else list(DEFAULT_RF_FEATURES)
    )
    cswi_limits = np.array(
        cswi_limits if cswi_limits is not None else [-0.5], dtype=np.float32
    )
    for name in feature_names:
        if name not in ds:
            raise KeyError(f"Missing feature: {name}")
    kwargs = dict(DEFAULT_RF_KWARGS)
    if rf_kwargs is not None:
        kwargs.update(rf_kwargs)
    feature_cube = build_feature_cube(ds, feature_names)
    nanflag = create_nanflag(ds, feature_names)
    if "qualityFlag" not in ds:
        raise KeyError("qualityFlag is required in the input dataset")
    if "vegFlag" not in ds or "NDVIFlag" not in ds or "posFlag" not in ds:
        raise KeyError("vegFlag, NDVIFlag and posFlag are required")
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
    out["RFflag"] = (
        ("lat", "lon", "timestamp", "CSWIlims"),
        np.zeros((n_lat, n_lon, n_time, n_limit), dtype=bool),
    )
    fill = np.full((n_lat, n_lon, n_time, n_limit), np.nan, dtype=np.float32)
    out["RTEM_WUE"] = (("lat", "lon", "timestamp", "CSWIlims"), fill.copy())
    out["RTEM_T"] = (("lat", "lon", "timestamp", "CSWIlims"), fill.copy())
    out["RTEM_E"] = (("lat", "lon", "timestamp", "CSWIlims"), fill.copy())
    out["feature_importances"] = (
        ("RFmod_vars", "CSWIlims"),
        np.full((n_feat, n_limit), np.nan, dtype=np.float32),
    )
    out["NumForestPoints"] = (
        ("CSWIlims",),
        np.full((n_limit,), np.nan, dtype=np.float32),
    )
    out["oob_scores"] = (("CSWIlims",), np.full((n_limit,), np.nan, dtype=np.float32))
    y_all = ds["WUEt"].values.astype(np.float32)
    for idx, limit in enumerate(cswi_limits):
        cur_flag = baseflag & (ds["CSWI"].values < limit) & ds["WUEtFlag"].values
        if baseline_training_mask is not None:
            if len(cswi_limits) != 1:
                raise ValueError("A reference mask requires a single CSWI threshold")
            if not np.array_equal(cur_flag, baseline_training_mask):
                raise ValueError("Training selection differs from the reference RFflag")
        out["RFflag"].values[:, :, :, idx] = cur_flag
        out["NumForestPoints"].values[idx] = float(cur_flag.sum())
        if curflag_dir is not None:
            os.makedirs(curflag_dir, exist_ok=True)
            np.save(os.path.join(curflag_dir, f"cur_flag_CSWI_{limit}.npy"), cur_flag)
            print(f"Training mask saved to {curflag_dir}, CSWI<{limit}")
        if cur_flag.sum() < min_train_samples:
            print(
                f"CSWI<{limit}: insufficient training samples; skipped. n={cur_flag.sum()}"
            )
            continue
        train_x = feature_cube[cur_flag]
        train_y = y_all[cur_flag]
        valid_y = np.isfinite(train_y) & (train_y > 0)
        train_x = train_x[valid_y]
        train_y = train_y[valid_y]
        if train_y.size < min_train_samples:
            print(
                f"CSWI<{limit}: insufficient positive target samples; skipped. n={train_y.size}"
            )
            continue
        forest = RandomForestRegressor(**kwargs)
        forest.fit(train_x, train_y)
        forest.predictor_names_ = list(feature_names)
        if model_path is not None:
            if len(cswi_limits) != 1:
                raise ValueError("A single model_path requires a single CSWI threshold")
            from pathlib import Path
            from joblib import dump

            target = Path(model_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            dump(forest, target, compress=1)
        print(f"CSWI<{limit}: random forest fitted. n={train_y.size}")
        if hasattr(forest, "oob_score_"):
            out["oob_scores"].values[idx] = np.float32(forest.oob_score_)
        out["feature_importances"].values[:, idx] = forest.feature_importances_.astype(
            np.float32
        )
        pred_mask = nanflag
        pred_x = feature_cube[pred_mask]
        if prediction_percentile is None:
            pred_wue = forest.predict(pred_x).astype(np.float32)
        else:
            pred_wue = predict_forest_percentile(
                forest,
                pred_x,
                percentile=float(prediction_percentile),
                chunk_size=prediction_chunk_size,
            )
        pred_grid = np.full((n_lat, n_lon, n_time), np.nan, dtype=np.float32)
        pred_grid[pred_mask] = pred_wue
        out["RTEM_WUE"].values[:, :, :, idx] = pred_grid
        t_grid = np.full_like(pred_grid, np.nan)
        valid = (
            np.isfinite(pred_grid)
            & (pred_grid > 0)
            & np.isfinite(ds["GPP"].values)
            & (ds["GPP"].values >= 0)
        )
        t_grid[valid] = ds["GPP"].values[valid] / pred_grid[valid]
        t_grid[~np.isfinite(t_grid) | (t_grid < 0)] = np.nan
        e_grid = ds["ET"].values.astype(np.float32) - t_grid
        out["RTEM_T"].values[:, :, :, idx] = t_grid.astype(np.float32)
        out["RTEM_E"].values[:, :, :, idx] = e_grid.astype(np.float32)
    out.attrs["features"] = ",".join(feature_names)
    out.attrs["PartRunTimestamp"] = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    out.attrs["PartType"] = os.path.basename(__file__)
    out.attrs["ModelType"] = "RandomForestRegressor"
    out["oob_scores"].attrs[
        "description"
    ] = "OOB R2 of the forest ensemble mean, not the P85 estimator"
    out.attrs["PredictionAggregation"] = (
        "mean"
        if prediction_percentile is None
        else f"tree_prediction_percentile_{prediction_percentile:g}"
    )
    if prediction_percentile is not None:
        out.attrs["PredictionPercentile"] = float(prediction_percentile)
    return out


if __name__ == "__main__":
    print("Import fit_predict_wuet from this module or run run_rtem.py.")
