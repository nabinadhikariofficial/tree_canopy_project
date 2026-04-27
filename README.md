# Tree Canopy Correction Project

This project predicts tree canopy cover from annual raster inputs using:

- a `RandomForestRegressor` baseline
- a `UNet` dense regressor
- a ViT-based dense regressor with optional Prithvi pretrained weights
- optional temporal consistency regularization across adjacent years

The current feature stack is fixed to 5 channels:

1. `nlcd`
2. `landcover`
3. `ndvi`
4. `lst`
5. `tmean`

## Project Flow

The intended workflow is:

1. Prepare and validate the yearly rasters.
2. Train a baseline model.
3. Train a deep model with or without temporal consistency.
4. Predict a full raster for a target year.
5. Evaluate predictions against the held-out canopy reference.

## Recommended Final-Project Configuration

Based on the current results, the strongest path for the final project is the U-Net family, not the Prithvi runs. The recommended settings are:

- use `--model unet`
- use temporal consistency across `2015..2022`
- use denser supervised patch extraction with `--train-stride 64`
- evaluate with `--eval-stride 64`
- keep train-time augmentation enabled
- predict full rasters with `--stride 64 --tta`

The training pipeline now also fits a validation-based calibration model automatically and stores it in `best.pt` plus `calibration.json`. Full-raster inference applies that calibration automatically when it is present.

Every new launcher-driven run now goes into a fresh timestamped folder using the current date and time, so old results are preserved by default.

## Expected Dataset Layout

```text
data/
  2015/
    2015_Tree_NLCD.tif
    2015_NLCD_LandCover.tif
    NDVI_2015_Albers_clip_resampled.tif
    LST_2015_Albers_clip_resampled.tif
    Tree_2015.tif
    Annual Prism Data/
      prism_tmean_us_30s_2015_clip_resampled.tif
  2016/
    ...
  2023/
    ...
```

Optional PRISM layers such as `ppt`, `tmin`, and `tmax` may exist, but the training code only uses `tmean`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies:

- `numpy`
- `rasterio`
- `scikit-learn`
- `matplotlib`
- `torch`
- `torchvision`
- `timm`

## 1. Prepare The Dataset

This validates the raster layout, records metadata, aligns labels when needed, and computes channel statistics from the training years.

```bash
python src/prepare_dataset.py \
  --data-root data \
  --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --output-dir runs/prepare_2023
```

Outputs:

- `runs/prepare_2023/manifest.json`
- `runs/prepare_2023/stats.json`
- aligned rasters under `runs/prepare_2023/aligned/`

## 2. Train The Random Forest Baseline

This uses only labeled years for supervision.

```bash
python src/train_rf.py \
  --data-root data \
  --train-years 2015 2021 \
  --test-year 2023 \
  --output-dir runs/rf_2023 \
  --sample-fraction 0.05
```

Outputs:

- `runs/rf_2023/rf.joblib`
- `runs/rf_2023/metrics.json`

## 3. Train Deep Models

The deep pipeline supports:

- `--train-years`: all years loaded into memory
- `--supervised-years`: years with labels used for supervised loss
- `--consistency-years`: years used to form temporal consistency pairs
- `--test-year`: held-out labeled evaluation year

### U-Net Baseline

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2021 \
  --test-year 2023 \
  --output-dir runs/unet_2023 \
  --model unet \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --consistency-weight 0.0
```

### U-Net With Consistency

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --output-dir runs/unet_consistency_2023 \
  --model unet \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --consistency-weight 0.2 \
  --change-gamma 3.0
```

### Prithvi / Geo-ViT

Use `--model geo_vit` with one of the local Prithvi checkpoints:

- `model_prithivi_weights/Prithvi_EO_V2_tiny_TL.pt`
- `model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt`

The code now auto-detects the matching ViT backbone for these checkpoints, so you do not need to set `--vit-name` manually.

### Prithvi Tiny Example

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --output-dir runs/prithvi_tiny_2023 \
  --model geo_vit \
  --pretrained-checkpoint model_prithivi_weights/Prithvi_EO_V2_tiny_TL.pt \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --consistency-weight 0.0
```

### Prithvi 100M Example

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --output-dir runs/prithvi_100m_2023 \
  --model geo_vit \
  --pretrained-checkpoint model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --consistency-weight 0.0
```

### Prithvi With Consistency

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --output-dir runs/prithvi_consistency_2023 \
  --model geo_vit \
  --pretrained-checkpoint model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --consistency-weight 0.2 \
  --change-gamma 3.0
```

Deep-model outputs include:

- `stats.json`
- `dataset_summary.json`
- `history.json`
- `best.pt`
- `test_metrics.json`

## 4. Predict A Full Raster

### Random Forest

```bash
python src/predict_raster_rf.py \
  --data-root data \
  --year 2023 \
  --model-path runs/rf_2023/rf.joblib \
  --output-path runs/rf_2023/pred_2023.tif \
  --reference-year 2015
```

### Deep Models

Use a trained deep checkpoint and its matching `stats.json`.

```bash
python src/predict_raster.py \
  --data-root data \
  --year 2023 \
  --checkpoint runs/prithvi_100m_2023/best.pt \
  --stats-path runs/prithvi_100m_2023/stats.json \
  --output-path runs/prithvi_100m_2023/pred_2023.tif \
  --patch-size 128 \
  --stride 96
```

Prediction rasters are written in canopy percent units `0..100`, with nodata preserved where no valid prediction is available.

## 5. Evaluate A Predicted Raster

Compare the predicted raster to the held-out label raster and the NLCD reference.

```bash
python src/evaluate_predictions.py \
  --pred-path runs/prithvi_100m_2023/pred_2023.tif \
  --target-path data/2023/Tree_2023.tif \
  --reference-path data/2023/2023_Tree_NLCD.tif \
  --output-dir runs/prithvi_100m_2023/eval_2023
```

Outputs:

- `metrics.txt`
- `bias_by_bin.png`
- `residual_hist.png`

## 6. Run The Full Experiment Grid

This runs:

- random forest baseline
- U-Net baseline
- U-Net with consistency
- Prithvi baseline
- Prithvi with consistency

```bash
python src/run_experiments.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --output-root runs/exp_2023 \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --prithvi-checkpoint model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt
```

Outputs:

- per-model subdirectories under `runs/exp_2023/`
- `runs/exp_2023/summary.json`

## Notes

- Labels are treated as high-resolution supervision, not absolute ground truth.
- Spatial leakage is reduced using block-based spatial splits instead of random pixels.
- The consistency term uses same-location patches across years and downweights pairs that show stronger input changes.
- Prithvi checkpoints are not plain timm ViT checkpoints; this repository now handles the shipped `tiny` and `100M` checkpoint formats directly.

## Verified In This Workspace

The following paths have been smoke-tested in this workspace:

- `python src/prepare_dataset.py ... --output-dir /tmp/tree_prepare_smoke`
- `python src/train_deep.py ... --model geo_vit --pretrained-checkpoint model_prithivi_weights/Prithvi_EO_V2_tiny_TL.pt --epochs 1`

The one-epoch Prithvi tiny training run completed successfully on the local `2015-2023` dataset and produced validation and test metrics.

## Recommended Reportable Experiments

Use this exact set for the final comparison table.

### 1. Prepare Dataset

```bash
python src/prepare_dataset.py \
  --data-root data \
  --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --reference-year 2015 \
  --output-dir runs/prepare_2023
```

### 2. Random Forest Baseline

Train:

```bash
python src/train_rf.py \
  --data-root data \
  --train-years 2015 2021 \
  --test-year 2023 \
  --reference-year 2015 \
  --block-size 256 \
  --seed 42 \
  --output-dir runs/rf_2023 \
  --sample-fraction 0.05
```

Predict full raster:

```bash
python src/predict_raster_rf.py \
  --data-root data \
  --year 2023 \
  --model-path runs/rf_2023/rf.joblib \
  --output-path runs/rf_2023/pred_2023.tif \
  --reference-year 2015
```

Evaluate full raster:

```bash
python src/evaluate_predictions.py \
  --pred-path runs/rf_2023/pred_2023.tif \
  --target-path data/2023/Tree_2023.tif \
  --reference-path data/2023/2023_Tree_NLCD.tif \
  --output-dir runs/rf_2023/eval_2023
```

### 3. U-Net Baseline

Train:

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2021 \
  --test-year 2023 \
  --reference-year 2015 \
  --output-dir runs/unet_2023 \
  --model unet \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --stride 128 \
  --block-size 256 \
  --num-workers 4 \
  --consistency-weight 0.0
```

Predict:

```bash
python src/predict_raster.py \
  --data-root data \
  --year 2023 \
  --checkpoint runs/unet_2023/best.pt \
  --stats-path runs/unet_2023/stats.json \
  --output-path runs/unet_2023/pred_2023.tif \
  --stride 96
```

Evaluate:

```bash
python src/evaluate_predictions.py \
  --pred-path runs/unet_2023/pred_2023.tif \
  --target-path data/2023/Tree_2023.tif \
  --reference-path data/2023/2023_Tree_NLCD.tif \
  --output-dir runs/unet_2023/eval_2023
```

### 4. U-Net With Temporal Consistency

Train:

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --reference-year 2015 \
  --output-dir runs/unet_consistency_2023 \
  --model unet \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --stride 128 \
  --block-size 256 \
  --num-workers 4 \
  --consistency-weight 0.2 \
  --change-gamma 3.0
```

Predict:

```bash
python src/predict_raster.py \
  --data-root data \
  --year 2023 \
  --checkpoint runs/unet_consistency_2023/best.pt \
  --stats-path runs/unet_consistency_2023/stats.json \
  --output-path runs/unet_consistency_2023/pred_2023.tif \
  --stride 96
```

Evaluate:

```bash
python src/evaluate_predictions.py \
  --pred-path runs/unet_consistency_2023/pred_2023.tif \
  --target-path data/2023/Tree_2023.tif \
  --reference-path data/2023/2023_Tree_NLCD.tif \
  --output-dir runs/unet_consistency_2023/eval_2023
```

### 5. Prithvi 100M Baseline

Train:

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --reference-year 2015 \
  --output-dir runs/prithvi_100m_2023 \
  --model geo_vit \
  --pretrained-checkpoint model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --stride 128 \
  --block-size 256 \
  --num-workers 4 \
  --consistency-weight 0.0
```

Predict:

```bash
python src/predict_raster.py \
  --data-root data \
  --year 2023 \
  --checkpoint runs/prithvi_100m_2023/best.pt \
  --stats-path runs/prithvi_100m_2023/stats.json \
  --output-path runs/prithvi_100m_2023/pred_2023.tif \
  --stride 96
```

Evaluate:

```bash
python src/evaluate_predictions.py \
  --pred-path runs/prithvi_100m_2023/pred_2023.tif \
  --target-path data/2023/Tree_2023.tif \
  --reference-path data/2023/2023_Tree_NLCD.tif \
  --output-dir runs/prithvi_100m_2023/eval_2023
```

### 6. Prithvi 100M With Temporal Consistency

Train:

```bash
python src/train_deep.py \
  --data-root data \
  --train-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --supervised-years 2015 2021 \
  --consistency-years 2015 2016 2017 2018 2019 2020 2021 2022 \
  --test-year 2023 \
  --reference-year 2015 \
  --output-dir runs/prithvi_consistency_2023 \
  --model geo_vit \
  --pretrained-checkpoint model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt \
  --epochs 40 \
  --batch-size 8 \
  --patch-size 128 \
  --stride 128 \
  --block-size 256 \
  --num-workers 4 \
  --consistency-weight 0.2 \
  --change-gamma 3.0
```

Predict:

```bash
python src/predict_raster.py \
  --data-root data \
  --year 2023 \
  --checkpoint runs/prithvi_consistency_2023/best.pt \
  --stats-path runs/prithvi_consistency_2023/stats.json \
  --output-path runs/prithvi_consistency_2023/pred_2023.tif \
  --stride 96
```

Evaluate:

```bash
python src/evaluate_predictions.py \
  --pred-path runs/prithvi_consistency_2023/pred_2023.tif \
  --target-path data/2023/Tree_2023.tif \
  --reference-path data/2023/2023_Tree_NLCD.tif \
  --output-dir runs/prithvi_consistency_2023/eval_2023
```

## Final Reporting Files

Use these for the comparison table:

- `runs/rf_2023/metrics.json`
- `runs/unet_2023/test_metrics.json`
- `runs/unet_consistency_2023/test_metrics.json`
- `runs/prithvi_100m_2023/test_metrics.json`
- `runs/prithvi_consistency_2023/test_metrics.json`

Use these for final map-quality analysis:

- `eval_2023/metrics.txt`
- `eval_2023/bias_by_bin.png`
- `eval_2023/residual_hist.png`

The primary final model is expected to be `runs/prithvi_consistency_2023`.
