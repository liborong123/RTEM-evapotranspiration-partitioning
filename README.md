# RTEM evapotranspiration partitioning

Code accompanying the manuscript “Evapotranspiration partitioning with machine learning and controls on vegetation transpiration in the Three-River Headwater Region”.

## Data layout

Extract the data archives and this code archive into the same parent directory:

```text
project/
  code/
  data/
    01_processed_inputs/
    02_final_RF_outputs/
    03_reference_products/
    04_validation/
    05_GAM/
    06_final_tables_figures/
    07_retained_sensitivity_experiment/
    08_supplementary/
  results/
```

Set `RTEM_DATA_ROOT` and `RTEM_RESULTS_ROOT` if these directories are elsewhere. Analysis and plotting scripts write to `results/`. Existing data filenames are supported without renaming the frozen files. `RTEM_RESULT_FILE` can specify the final RTEM NetCDF explicitly.

## RTEM

The root files retain the repository entry points: `data_preprocessing.py`, `rtem_algorithm.py`, and `run_rtem.py`. The model predicts daily WUEt from the manuscript's 11 predictors. Parameters are 300 trees, square-root feature sampling, minimum leaf size 3, random seed 42, NDVI90 > 0.55, CSWI < -0.5, and the 85th percentile of individual-tree predictions. All parameters remain explicit in the code. SWC and static soil attributes are not RF predictors. SWC is used in the environmental analysis; CSWI is a separate RF input.

```text
python run_rtem.py --input ../data/01_processed_inputs/RF_ready/current_preprocessed_template.nc --output ../results/rtem
```

The prepared NetCDF includes the actual predictor arrays and screening flags. `preprocessing/compute_initial_labels.py` calculates the initial uWUE slope and labels. `preprocessing/compute_et.py` and `compute_vpd.py` preserve the SFE and VPD equations, taking arrays in physical units. Their command-line input handling has been organized for this release; it does not imply that the original provider-to-processed-data chain has been independently rerun. The data archive supplies the processed ET and VPD actually used. `screen_wuet.py` contains the MAD/upper-limit screening functions.

## Evaluation and figures

`analysis/calculate_temporal_metrics.py` uses the final paired regional time series and the manuscript's April–September season. It does not compare alternative season definitions. `analysis/calculate_site_ratios.py` implements the reported field-site monthly aggregation. Figure scripts reproduce the final model's product comparisons, spatial patterns, grouped GAM figures, framework and supplementary RF importance. Supplementary scripts use the reported vegetation-cover definition and individual environmental ranges.

## GAM

Install the analysis dependencies, including pyGAM 0.12.0 and scikit-learn 1.7.2, in a separate environment as needed. Run `python analysis/verify_gam.py` to check the archived model. To fit the analysis again, run `python analysis/fit_gam.py`. Group-wise standardization and PCA use training pixels only, the split is 80:20 with seed 42, and importance is the normalized decline in training explained-deviance after removing and refitting each group. These are associations with RTEM-estimated T.

## Availability and scope

Input datasets originate from public sources. Original source citations and access locations are provided with the data archive; third-party licences and citations continue to apply. Code and data require permanent repository records and actual DOI citations for resubmission.

The manuscript's Figure 6 parameter-sensitivity results are preserved in `data/07_retained_sensitivity_experiment/`; the original complete experiment-generation and plotting scripts have not been located. Figure 3 was assembled in plotting software; its original point-pair table/project has not been located. No replacement observations or experiment outcomes have been generated. These limitations are not resolved merely by packaging the final scripts.

Existing filenames inside data and saved-model metadata are retained for traceability.

## License and citation

The software is released under the MIT License. Third-party libraries and datasets retain their respective licences. Cite the Zenodo software record and the associated research article when available. `CITATION.cff` provides the software authors and version; `.zenodo.json` supplies the archive metadata.
