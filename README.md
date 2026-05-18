# RTEM-TRHR Transpiration Workflow

This repository contains the Python scripts used to preprocess raster datasets and implement the Random Forest-based WUEt extrapolation component of the Regional Transpiration Estimation Model (RTEM). The workflow was developed for regional vegetation transpiration estimation in the Three-River Headwaters Region (TRHR).

## Repository structure

```text
RTEM_code_clean/
├── data_preprocessing.py   # Data loading, quality control, derived variables, and NetCDF assembly
├── rtem_algorithm.py       # Random Forest-based WUEt extrapolation and T/E estimation
├── run_rtem.py             # Main workflow script
├── requirements.txt        # Python dependencies
└── README.md               # Usage notes
```

## Workflow overview

The workflow consists of two main steps:

1. **Data preprocessing** (`data_preprocessing.py`)
   - Loads daily meteorological, carbon-flux, vegetation, and soil-property rasters.
   - Converts NoData values to `NaN`.
   - Builds a common set of dates across dynamic variables.
   - Computes auxiliary variables including CSWI, vegetation-available radiation (`Qv`), GPP gradients, potential radiation (`Rg_pot`), and seasonal time variables.
   - Creates quality-control and training-sample flags.
   - Stores the processed dataset as an `xarray.Dataset`.

2. **WUEt extrapolation and transpiration estimation** (`rtem_algorithm.py`)
   - Selects training samples using vegetation, quality-control, positivity, NDVI90%, WUEt, and CSWI filters.
   - Trains a `RandomForestRegressor` to predict WUEt from meteorological, radiation, vegetation, and soil predictors.
   - Predicts spatially continuous WUEt.
   - Estimates transpiration as `T = GPP / WUEt` and soil evaporation as `E = ET - T`.
   - Exports feature importances and out-of-bag scores.

## Input data organization

Before running the scripts, update the paths in `run_rtem.py`. The default script assumes a folder structure similar to:

```text
meteorology_root/
├── prec/
├── ET/
├── GPP_PML/
├── VPD/
├── LAI/
├── wind/
├── Rn/
├── temp/
├── shum/
├── uWUEt/
├── SWC/
├── pres/
├── srad/
└── NDVI/

soil_root/
└── *.tif

vegetation_root/
└── *.tif
```

Dynamic raster files should contain a date in the filename, with the date formatted as `YYYY-MM-DD`. For example:

```text
ET_2015-01-01.tif
GPP_2015-01-01.tif
VPD_2015-01-01.tif
```

All rasters should already be spatially aligned before running the workflow. The preprocessing script does not perform reprojection or resampling.

## Installation

Create a clean Python environment and install dependencies:

```bash
conda create -n rtem python=3.9
conda activate rtem
conda install -c conda-forge gdal
pip install -r requirements.txt
```

GDAL installation is usually more reliable through `conda-forge` than through `pip`, especially on Windows.

## How to run

1. Open `run_rtem.py`.
2. Modify the following paths and configuration values:

```python
meteorology_root = r"G:\caculate_T"
soil_root = r"G:\soil property\...\temp_origin"
vegetation_root = r"G:\veg"
out_dir = Path(r"G:\tera_output_v2\uWUE")
```

3. Run the workflow:

```bash
python run_rtem.py
```

## Main outputs

The workflow writes the following files to the output directory:

```text
preprocessed_dataset.nc    # Preprocessed dynamic/static variables and flags
rtem_result.nc             # RTEM_WUE, RTEM_T, RTEM_E, and diagnostic variables
feature_importances.csv    # Random Forest predictor importances
oob_scores.csv             # Out-of-bag scores for each CSWI threshold
```

## Key parameters

Important parameters are configured in `run_rtem.py` and `DataConfig`:

- `start_date`, `end_date`: time range used in the analysis.
- `precip_threshold_mm_day`: precipitation threshold used in preprocessing.
- `ndvi90_threshold`: NDVI90% threshold used to identify high-vegetation-cover training areas.
- `vegetation_classes`: target vegetation classes retained by the vegetation mask.
- `cswi_limits`: CSWI thresholds used for selecting training samples.
- `rf_kwargs`: Random Forest hyperparameters.
- `min_train_samples`: minimum number of valid training samples required for model fitting.

## Notes

- Static soil variables are stored as 2D arrays and are broadcast to 3D arrays only inside the modeling module.
- The optional `rf_flag_output_dir` argument in `fit_predict_wuet()` can be used to save RF training masks for debugging. It is disabled by default.
- The scripts use `NaN` internally for invalid raster values. Values less than or equal to `-9000` are also treated as invalid.
- Large input rasters and NetCDF outputs should not be committed directly to GitHub. Use a data repository or provide data availability information in the manuscript.

## Citation

If you use this code, please cite the associated manuscript:

Li, B., Huang, Y., Li, B., Chen, Z., Zhang, S. Evapotranspiration partitioning with machine learning and controls on vegetation transpiration in the Three-River Headwater Region.

## Data and code availability statement

The original remote sensing and environmental datasets used in this study are publicly available from the data repositories cited in the manuscript. The processed datasets and Python scripts used for model implementation, statistical analysis, and visualization are available from the corresponding author upon reasonable request.
