"""Run the 11-predictor RTEM model on aligned, prepared NetCDF inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing import load_model_inputs
from rtem_algorithm import DEFAULT_RF_FEATURES, DEFAULT_RF_KWARGS, fit_predict_wuet

MANUSCRIPT_SYMBOLS = [
    "T_a",
    "q_a",
    "u",
    "dGPP/dt",
    "Rg_pot",
    "dRg_pot/dy",
    "dRg_pot/dt",
    "CSWI",
    "LAI",
    "P_a",
    "R_n",
]


def export_daily_transpiration(result, reference_raster, folder):
    """Write daily T with the unchanged reference-raster georeferencing."""
    from osgeo import gdal

    gdal.UseExceptions()
    reference = gdal.Open(str(reference_raster))
    if reference is None:
        raise ValueError("Cannot open the reference raster")
    transform, projection = reference.GetGeoTransform(), reference.GetProjection()
    if (reference.RasterYSize, reference.RasterXSize) != (
        result.sizes["lat"],
        result.sizes["lon"],
    ):
        raise ValueError("Reference raster and result shapes differ")
    reference = None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for index, stamp in enumerate(result.timestamp.values):
        values = result.RTEM_T.isel(timestamp=index).values
        target = folder / f"T_RTEM11_P85_{str(stamp)[:10]}.tif"
        raster = gdal.GetDriverByName("GTiff").Create(
            str(target),
            values.shape[1],
            values.shape[0],
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES"],
        )
        raster.SetGeoTransform(transform)
        raster.SetProjection(projection)
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(
            np.where(np.isfinite(values) & (values >= 0), values, -9999).astype(
                "float32"
            )
        )
        band.FlushCache()
        band = None
        raster = None


def run(input_path, output_dir, n_jobs=16, chunk_size=100000, reference_raster=None):
    """Fit the source model, verify its sample mask, and export its estimates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, reference_mask, source_metadata = load_model_inputs(input_path)
    parameters = dict(DEFAULT_RF_KWARGS, n_jobs=n_jobs)
    estimates = fit_predict_wuet(
        inputs,
        feature_names=DEFAULT_RF_FEATURES,
        cswi_limits=[-0.5],
        rf_kwargs=parameters,
        prediction_percentile=85,
        prediction_chunk_size=chunk_size,
        model_path=str(output_dir / "rf_model_11vars.joblib"),
        baseline_training_mask=reference_mask,
    )
    if (
        not np.isfinite(estimates.NumForestPoints.values[0])
        or estimates.NumForestPoints.values[0] < 500
    ):
        raise ValueError("Fewer than 500 training-mask samples; no model was fitted")
    mask = estimates.RFflag.isel(CSWIlims=0).values
    labels = inputs.WUEt.values[mask].astype("float32")
    positive = np.isfinite(labels) & (labels > 0)
    if positive.sum() < 500:
        raise ValueError("Fewer than 500 positive target samples; no model was fitted")
    result = estimates[
        ["RTEM_WUE", "RTEM_T", "RTEM_E", "RFflag", "feature_importances"]
    ].isel(CSWIlims=0, drop=True)
    result.attrs.update(
        {
            "features": ",".join(DEFAULT_RF_FEATURES),
            "PredictionPercentile": 85.0,
            "GPP_source": "FluxSat",
            "Initial_uWUE_ET_source": "self-calculated SFE ET",
            "TrainingSamples": int(mask.sum()),
            "OOB_score": float(estimates.oob_scores.values[0]),
            "Experiment": "11 predictors; source labels and screening flags retained",
        }
    )
    result.to_netcdf(
        output_dir / "rtem_result.nc",
        engine="netcdf4",
        encoding={name: {"zlib": True, "complevel": 1} for name in result.data_vars},
    )
    importance = estimates.feature_importances.isel(CSWIlims=0).values
    pd.DataFrame(
        {
            "feature": DEFAULT_RF_FEATURES,
            "manuscript_symbol": MANUSCRIPT_SYMBOLS,
            "importance": importance,
            "importance_percent": 100 * importance,
        }
    ).to_csv(output_dir / "feature_importances.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"CSWI_threshold": [-0.5], "ensemble_mean_oob_R2": estimates.oob_scores.values}
    ).to_csv(
        output_dir / "oob_scores.csv",
        index=False,
    )
    audit = {
        "input_file": Path(input_path).name,
        "features": DEFAULT_RF_FEATURES,
        "feature_count": len(DEFAULT_RF_FEATURES),
        "RF_parameters": parameters,
        "prediction_percentile": 85,
        "training_samples": int(positive.sum()),
        "training_mask_samples": int(mask.sum()),
        "reference_RFflag_available": reference_mask is not None,
        "reference_RFflag_identical": (
            bool(np.array_equal(mask, reference_mask))
            if reference_mask is not None
            else None
        ),
        "training_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "training_labels_sha256": hashlib.sha256(
            labels[positive].tobytes()
        ).hexdigest(),
        "grid_shape": [inputs.sizes[axis] for axis in ("lat", "lon", "timestamp")],
        "first_date": str(inputs.timestamp.values[0])[:10],
        "last_date": str(inputs.timestamp.values[-1])[:10],
        "source_ET_description": source_metadata.get("Initial_uWUE_ET_source"),
        "source_GPP_description": source_metadata.get("GPP_source"),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if reference_raster is not None:
        export_daily_transpiration(result, reference_raster, output_dir / "RTEM_T")
    print(f"Saved results for {inputs.sizes['timestamp']} dates to {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Prepared input NetCDF file"
    )
    parser.add_argument("--output", required=True, type=Path, help="Result directory")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument(
        "--reference-raster",
        type=Path,
        help="Optional GeoTIFF reference for daily T export",
    )
    args = parser.parse_args()
    run(args.input, args.output, args.n_jobs, args.chunk_size, args.reference_raster)


if __name__ == "__main__":
    main()
