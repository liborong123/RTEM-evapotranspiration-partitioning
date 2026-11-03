# RTEM regional transpiration estimation

Python implementation of the 11-predictor random forest used to estimate regional vegetation transpiration in the Three-River Headwater Region.

## Repository structure

```text
RTEM-evapotranspiration-partitioning/
├── data_preprocessing.py   # Read aligned input arrays, existing labels and screening flags
├── rtem_algorithm.py       # Fit the random forest and calculate tree-prediction percentiles
├── run_rtem.py             # Command-line entry point and output export
├── requirements.txt
└── README.md
```

## Model configuration

The model predicts daily transpiration-related water-use efficiency using these eleven inputs, in order:

| Input variable | Description |
| --- | --- |
| `temp` | Air temperature |
| `shum` | Specific humidity |
| `wind` | Wind speed |
| `GPPgradtime` | Time derivative of smoothed gross primary productivity |
| `Rg_pot` | Potential incoming radiation |
| `Rgpot_grady` | Latitudinal gradient of potential radiation |
| `Rgpot_gradtime` | Temporal gradient of potential radiation |
| `CSWI` | Conservative surface wetness index |
| `LAI` | Leaf area index |
| `pres` | Air pressure |
| `Rn` | Net radiation |

The forest uses 300 trees, square-root feature sampling, a minimum leaf size of three, and random seed 42. Predictions are the pointwise 85th percentile of the individual-tree predictions. Transpiration is calculated as `GPP / WUEt`; evaporation is `ET - T`.

Static soil properties, vapor pressure deficit, vegetation-available radiation and soil water content are not predictor columns in this model. Vapor pressure deficit may be involved in constructing the original water-use-efficiency labels. Existing screening flags are sample-selection controls, not additional predictor columns.

## Input preparation

The source experiment reused the aligned SFE evapotranspiration, FluxSat gross primary productivity, water-use-efficiency labels and quality-control flags from an existing NetCDF dataset. This repository implements that model stage. It does not download the original products or rebuild SFE estimates, uWUE labels or GAM analyses from raw observations.

Provide a NetCDF file with dimensions `(lat, lon, timestamp)` containing all eleven inputs above and:

- `ET`, `GPP`, `WUEt`;
- `qualityFlag`, `vegFlag`, `NDVIFlag`, `posFlag`, `WUEtFlag`.

Use the units, grid, dates, missing-value conventions and label definitions described in the manuscript. `NDVIFlag` represents the `NDVI90% > 0.55` training criterion. Dry training dates additionally require `CSWI < -0.5`. Finite labels and the source WUEt screening flag are retained exactly as in the original experiment. If the file includes a reference `RFflag`, the program verifies that the selected training mask is identical before fitting.

The input labels and flags must come from the intended SFE-ET/FluxSat-GPP experiment. Changing the filename or variable names alone does not establish their provenance. The program reports available source metadata in `run_metadata.json`.

## Installation

```bash
python -m pip install -r requirements.txt
```

GDAL is only needed for optional daily GeoTIFF export. When using Conda, install it from conda-forge in the same environment:

```bash
conda install -c conda-forge gdal
```

## Run

```bash
python run_rtem.py --input model_inputs.nc --output results --n-jobs 16
```

For daily GeoTIFF output, supply an aligned reference raster:

```bash
python run_rtem.py --input model_inputs.nc --output results --reference-raster reference.tif
```

`--chunk-size` controls the number of samples processed together when calculating tree percentiles. It changes memory use, not the percentile definition. Select `--n-jobs` to suit the available CPU resources. Large regional datasets require substantial RAM because the original workflow constructs a complete feature cube.

## Outputs

- `rtem_result.nc`: predicted water-use efficiency, transpiration, evaporation, training mask and feature importances.
- `rf_model_11vars.joblib`: fitted forest and predictor names.
- `feature_importances.csv`: normalized mean decrease in impurity for all eleven inputs.
- `oob_scores.csv`: the forest's ensemble-mean out-of-bag R-squared. This diagnostic is not validation of the P85 estimator or an independent transpiration observation.
- `run_metadata.json`: model settings, array dimensions, dates and hashes of the training mask and positive labels.
- `RTEM_T/`: optional daily transpiration GeoTIFFs.

Model outputs are not clipped to enforce `T <= ET`; negative residual evaporation can therefore occur and should be examined when evaluating partitioning. The scripts preserve the source prediction-domain requirement that the original WUEt value is finite. Figures comparing products, seasonal-window analyses and GAM attribution are separate research analyses and are not run by this entry point.

## Data and code availability

The original input datasets are openly available from the providers and repositories cited in the manuscript. The related model scripts are provided in this repository. Dataset descriptions, source references and the preparation of the aligned inputs are documented in the manuscript.

## Citation

Li, B., Huang, Y., Li, B., Chen, Z., and Zhang, S. *Evapotranspiration partitioning with machine learning and controls on vegetation transpiration in the Three-River Headwater Region.*
