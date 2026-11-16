# Script inventory

| Script | Purpose |
|---|---|
| `project_paths.py` | Resolve input and output directories; preserve archived inputs. |
| `data_preprocessing.py` | Read prepared predictor arrays and screening flags. |
| `rtem_algorithm.py` | Fit the manuscript RF model and estimate WUEt, T and E. |
| `run_rtem.py` | Run RTEM and export the fitted model, rasters and metadata. |
| `preprocessing/compute_initial_labels.py` | Calculate the uWUE slope, WUEt labels and initial T. |
| `preprocessing/compute_et.py` | Evaluate the SFE ET equation using physical-unit arrays. |
| `preprocessing/compute_vpd.py` | Calculate VPD from temperature, specific humidity and pressure. |
| `preprocessing/screen_wuet.py` | Apply the 15-day MAD and WUEt upper-limit screening rules. |
| `analysis/prepare_environment.py` | Combine final mean T with preserved environmental layers. |
| `analysis/fit_gam.py` | Fit grouped PCA/GAM and compute drop-group importance (Figure 8). |
| `analysis/verify_gam.py` | Check saved GAM predictions, metrics and group importance. |
| `analysis/calculate_temporal_metrics.py` | Calculate final-product agreement for all dates and April-September (Figure 5). |
| `analysis/calculate_site_ratios.py` | Calculate monthly T/ET and the seven-month field-site summary. |
| `figures/plot_product_comparison.py` | Calculate and plot the Figure 4 product comparisons. |
| `figures/plot_temporal_comparison.py` | Plot regional T and T/ET time series (Figure 5). |
| `figures/plot_spatial_and_gam.py` | Plot spatial/environmental patterns and GAM results (Figures 7 and 8). |
| `figures/plot_framework.py` | Draw the RTEM workflow (Figure 2). |
| `figures/plot_rf_importance.py` | Plot final RF predictor importance (Figure S2). |
| `supplementary/calculate_cover_area.py` | Calculate full-boundary vegetation-cover area fractions (Figure S1). |
| `supplementary/calculate_environment_support.py` | Calculate individual-variable environmental range support (Figure S1). |
| `supplementary/plot_environment_support.py` | Plot vegetation-cover classes and environmental support (Figure S1). |
