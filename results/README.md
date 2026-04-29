# Results Assets

This folder contains the figures and tables referenced in the technical report draft.

## Figures

- `figures/figure_01_workflow_diagram.png`: end-to-end workflow diagram.
- `figures/figure_02_input_stack_sample.png`: sample 2023 crop of the five input channels.
- `figures/figure_03_map_comparison_2023.png`: qualitative 2023 target / baseline / model comparison.
- `figures/figure_04_edge_artifact_fix.png`: old border artifact versus corrected inference output.
- `figures/figure_05_metrics_bar_chart.png`: RMSE and R2 comparison across model families for `20260427_152647`.
- `figures/figure_06_final_unet_scatter.png`: scatter plot of final U-Net predictions versus target canopy.
- `figures/figure_07_learning_evolution_2015_2023.png`: yearly timeline showing supervised years, temporal-only context years, and the held-out 2023 test year.
- `figures/figure_08_run_progression.png`: performance progression across the major run generations from `runs 04_24` to `20260427_152647`.
- `figures/figure_09_yearly_canopy_evolution_2015_2023.png`: year-by-year NLCD canopy crop sequence from 2015 to 2023.
- `figures/figure_10_training_history_final_unet.png`: training and validation progress across epochs for the final U-Net models.

## Tables

- `tables/table_01_dataset_year_availability.{csv,md}`
- `tables/table_02_model_configuration_summary.{csv,md}`
- `tables/table_03_historical_run_progression.{csv,md}`
- `tables/table_04_final_consolidated_metrics_20260427_152647.{csv,md}`
- `tables/table_05_landcover_ablation.{csv,md}`
- `tables/table_06_limitations_and_future_work.{csv,md}`

These assets were generated from the actual rasters and metric files in `runs/`.
