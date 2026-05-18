# -*- coding: utf-8 -*-
"""Run the RTEM preprocessing and random-forest WUEt extrapolation workflow.

Outputs
-------
1. Preprocessed dataset: preprocessed_dataset.nc
2. Model-result dataset: rtem_result.nc
3. Feature-importance table: feature_importances.csv
4. Out-of-bag score table: oob_scores.csv

Usage
-----
python run_rtem.py

Edit the path, feature, and model-parameter settings in ``main()`` before use.
"""

from __future__ import annotations

from pathlib import Path

from data_preprocessing import DataConfig, load_all_data
from rtem_algorithm import fit_predict_wuet


def export_diagnostics(ds_result, out_dir: Path) -> None:
    """Export feature importances and out-of-bag scores."""
    # Feature importances
    fi = ds_result["feature_importances"].to_pandas()
    fi.to_csv(out_dir / "feature_importances.csv", encoding="utf-8-sig")

    # Out-of-bag scores
    oob = ds_result["oob_scores"].to_pandas()
    oob.to_csv(out_dir / "oob_scores.csv", encoding="utf-8-sig")


def main() -> None:
    # ==============================
    # 1. Path and data configuration
    # ==============================
    cfg = DataConfig(
        meteorology_root=r"G:\caculate_T",
        soil_root=r"G:\soil_property\temp_origin",
        vegetation_root=r"G:\veg",
        start_date="2015-01-01",
        end_date="2020-12-31",
        precip_threshold_mm_day=5.0,
        ndvi90_threshold=0.55,
        vegetation_classes=(10,),
        soil_var_patterns={
            "bd": ("bd1530_1km",),
            "clay": ("btcly1530_1km",),
            "silt": ("btslt1530_1km",),
            "sand": ("btsnd1530_1km",),
            "CEC": ("cec1530_1km",),
            "cf": ("cf1530_1km",),
            "pH": ("ph1530_1km",),
            "SOC": ("soc1530_1km",),
            "texcls": ("texcls1530_1km",),
            "thickness": ("thickness_1km",),
            "tk": ("tk1530_1km",),
            "tn": ("tn1530_1km",),
            "tp": ("tp1530_1km",),
        }
    )

    # Output directory
    out_dir = Path(r"G:\rtem_output\uWUE")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 2. Data loading and preprocessing
    # ==============================
    print("=" * 80)
    print("[1/3] Loading and preprocessing data...")
    ds = load_all_data(cfg)
    print(ds)

    preprocessed_nc = out_dir / "preprocessed_dataset.nc"
    ds.to_netcdf(preprocessed_nc)
    print(f"Preprocessed dataset saved to: {preprocessed_nc}")

    # ==============================
    # 3. Model fitting and prediction
    # ==============================
    print("=" * 80)
    print("[2/3] Training random forest and predicting WUEt / T / E...")

    feature_names = [
        "Qv", "temp", "shum", "wind", "VPD",
        "GPPgradtime", "Rg_pot", "Rgpot_grady", "Rgpot_gradtime",
        "CSWI", "LAI",
        "SWC", "pres", "Rn",
        "pH", "SOC", "CEC",
        "clay", "silt", "sand",
        "bd", "cf", "thickness",
        "tn"
    ]


    rf_kwargs = {
        "n_estimators": 300,
        "oob_score": True,
        "max_features": "sqrt",
        "min_samples_leaf": 3,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 1,
    }

    ds_result = fit_predict_wuet(
        ds=ds,
        feature_names=feature_names,
        cswi_limits=[-0.5],
        rf_kwargs=rf_kwargs,
        min_train_samples=500,
    )

    # ==============================
    # 4. Save outputs
    # ==============================
    print("=" * 80)
    print("[3/3] Saving outputs...")
    result_nc = out_dir / "rtem_result.nc"
    ds_result.to_netcdf(result_nc)
    export_diagnostics(ds_result, out_dir)

    print(f"Result dataset saved to: {result_nc}")
    print(f"Feature importances saved to: {out_dir / 'feature_importances.csv'}")
    print(f"OOB scores saved to: {out_dir / 'oob_scores.csv'}")
    print("Workflow completed.")


if __name__ == "__main__":
    main()
