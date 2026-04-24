#!/bin/bash
#SBATCH --job-name=tree-canopy-full
#SBATCH --partition=a30_normal_q
#SBATCH --account=cs5814
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dl}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_DIR/runs}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs}"

REFERENCE_YEAR="${REFERENCE_YEAR:-2015}"
TEST_YEAR="${TEST_YEAR:-2023}"

TRAIN_YEARS=(2015 2016 2017 2018 2019 2020 2021 2022)
SUPERVISED_YEARS=(2015 2021)
CONSISTENCY_YEARS=(2015 2016 2017 2018 2019 2020 2021 2022)

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATCH_SIZE="${PATCH_SIZE:-128}"
STRIDE="${STRIDE:-128}"
INFER_STRIDE="${INFER_STRIDE:-96}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

RF_SAMPLE_FRACTION="${RF_SAMPLE_FRACTION:-0.05}"
PRITHVI_CKPT="${PRITHVI_CKPT:-$PROJECT_DIR/model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt}"

mkdir -p "$LOG_ROOT" "$RUNS_ROOT"

cd "$PROJECT_DIR"

module load Miniconda3
source activate "$CONDA_ENV_NAME"

PYTHON_BIN="$(command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python not found after activating conda env: $CONDA_ENV_NAME" >&2
  exit 1
fi

echo "Project dir: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "Conda env: $CONDA_ENV_NAME"
echo "Data root: $DATA_ROOT"
echo "Runs root: $RUNS_ROOT"
echo "Reference year: $REFERENCE_YEAR"
echo "Test year: $TEST_YEAR"

"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("mps available:", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
PY

join_by_space() {
  local IFS=" "
  echo "$*"
}

TRAIN_YEARS_STR="$(join_by_space "${TRAIN_YEARS[@]}")"
SUPERVISED_YEARS_STR="$(join_by_space "${SUPERVISED_YEARS[@]}")"
CONSISTENCY_YEARS_STR="$(join_by_space "${CONSISTENCY_YEARS[@]}")"

echo "== Prepare dataset =="
"$PYTHON_BIN" src/prepare_dataset.py \
  --data-root "$DATA_ROOT" \
  --years $TRAIN_YEARS_STR "$TEST_YEAR" \
  --train-years $TRAIN_YEARS_STR \
  --reference-year "$REFERENCE_YEAR" \
  --output-dir "$RUNS_ROOT/prepare_2023"

echo "== Random Forest baseline =="
"$PYTHON_BIN" src/train_rf.py \
  --data-root "$DATA_ROOT" \
  --train-years $SUPERVISED_YEARS_STR \
  --test-year "$TEST_YEAR" \
  --reference-year "$REFERENCE_YEAR" \
  --block-size "$BLOCK_SIZE" \
  --seed "$SEED" \
  --output-dir "$RUNS_ROOT/rf_2023" \
  --sample-fraction "$RF_SAMPLE_FRACTION"

"$PYTHON_BIN" src/predict_raster_rf.py \
  --data-root "$DATA_ROOT" \
  --year "$TEST_YEAR" \
  --model-path "$RUNS_ROOT/rf_2023/rf.joblib" \
  --output-path "$RUNS_ROOT/rf_2023/pred_${TEST_YEAR}.tif" \
  --reference-year "$REFERENCE_YEAR"

"$PYTHON_BIN" src/evaluate_predictions.py \
  --pred-path "$RUNS_ROOT/rf_2023/pred_${TEST_YEAR}.tif" \
  --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
  --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
  --output-dir "$RUNS_ROOT/rf_2023/eval_${TEST_YEAR}"

echo "== U-Net baseline =="
"$PYTHON_BIN" src/train_deep.py \
  --data-root "$DATA_ROOT" \
  --train-years $TRAIN_YEARS_STR \
  --supervised-years $SUPERVISED_YEARS_STR \
  --consistency-years 2015 2021 \
  --test-year "$TEST_YEAR" \
  --reference-year "$REFERENCE_YEAR" \
  --output-dir "$RUNS_ROOT/unet_2023" \
  --model unet \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --block-size "$BLOCK_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --consistency-weight 0.0

"$PYTHON_BIN" src/predict_raster.py \
  --data-root "$DATA_ROOT" \
  --year "$TEST_YEAR" \
  --checkpoint "$RUNS_ROOT/unet_2023/best.pt" \
  --stats-path "$RUNS_ROOT/unet_2023/stats.json" \
  --output-path "$RUNS_ROOT/unet_2023/pred_${TEST_YEAR}.tif" \
  --stride "$INFER_STRIDE"

"$PYTHON_BIN" src/evaluate_predictions.py \
  --pred-path "$RUNS_ROOT/unet_2023/pred_${TEST_YEAR}.tif" \
  --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
  --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
  --output-dir "$RUNS_ROOT/unet_2023/eval_${TEST_YEAR}"

echo "== U-Net + consistency =="
"$PYTHON_BIN" src/train_deep.py \
  --data-root "$DATA_ROOT" \
  --train-years $TRAIN_YEARS_STR \
  --supervised-years $SUPERVISED_YEARS_STR \
  --consistency-years $CONSISTENCY_YEARS_STR \
  --test-year "$TEST_YEAR" \
  --reference-year "$REFERENCE_YEAR" \
  --output-dir "$RUNS_ROOT/unet_consistency_2023" \
  --model unet \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --block-size "$BLOCK_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --consistency-weight 0.2 \
  --change-gamma 3.0

"$PYTHON_BIN" src/predict_raster.py \
  --data-root "$DATA_ROOT" \
  --year "$TEST_YEAR" \
  --checkpoint "$RUNS_ROOT/unet_consistency_2023/best.pt" \
  --stats-path "$RUNS_ROOT/unet_consistency_2023/stats.json" \
  --output-path "$RUNS_ROOT/unet_consistency_2023/pred_${TEST_YEAR}.tif" \
  --stride "$INFER_STRIDE"

"$PYTHON_BIN" src/evaluate_predictions.py \
  --pred-path "$RUNS_ROOT/unet_consistency_2023/pred_${TEST_YEAR}.tif" \
  --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
  --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
  --output-dir "$RUNS_ROOT/unet_consistency_2023/eval_${TEST_YEAR}"

echo "== Prithvi 100M baseline =="
"$PYTHON_BIN" src/train_deep.py \
  --data-root "$DATA_ROOT" \
  --train-years $TRAIN_YEARS_STR \
  --supervised-years $SUPERVISED_YEARS_STR \
  --consistency-years $CONSISTENCY_YEARS_STR \
  --test-year "$TEST_YEAR" \
  --reference-year "$REFERENCE_YEAR" \
  --output-dir "$RUNS_ROOT/prithvi_100m_2023" \
  --model geo_vit \
  --pretrained-checkpoint "$PRITHVI_CKPT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --block-size "$BLOCK_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --consistency-weight 0.0

"$PYTHON_BIN" src/predict_raster.py \
  --data-root "$DATA_ROOT" \
  --year "$TEST_YEAR" \
  --checkpoint "$RUNS_ROOT/prithvi_100m_2023/best.pt" \
  --stats-path "$RUNS_ROOT/prithvi_100m_2023/stats.json" \
  --output-path "$RUNS_ROOT/prithvi_100m_2023/pred_${TEST_YEAR}.tif" \
  --stride "$INFER_STRIDE"

"$PYTHON_BIN" src/evaluate_predictions.py \
  --pred-path "$RUNS_ROOT/prithvi_100m_2023/pred_${TEST_YEAR}.tif" \
  --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
  --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
  --output-dir "$RUNS_ROOT/prithvi_100m_2023/eval_${TEST_YEAR}"

echo "== Prithvi 100M + consistency =="
"$PYTHON_BIN" src/train_deep.py \
  --data-root "$DATA_ROOT" \
  --train-years $TRAIN_YEARS_STR \
  --supervised-years $SUPERVISED_YEARS_STR \
  --consistency-years $CONSISTENCY_YEARS_STR \
  --test-year "$TEST_YEAR" \
  --reference-year "$REFERENCE_YEAR" \
  --output-dir "$RUNS_ROOT/prithvi_consistency_2023" \
  --model geo_vit \
  --pretrained-checkpoint "$PRITHVI_CKPT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --block-size "$BLOCK_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --consistency-weight 0.2 \
  --change-gamma 3.0

"$PYTHON_BIN" src/predict_raster.py \
  --data-root "$DATA_ROOT" \
  --year "$TEST_YEAR" \
  --checkpoint "$RUNS_ROOT/prithvi_consistency_2023/best.pt" \
  --stats-path "$RUNS_ROOT/prithvi_consistency_2023/stats.json" \
  --output-path "$RUNS_ROOT/prithvi_consistency_2023/pred_${TEST_YEAR}.tif" \
  --stride "$INFER_STRIDE"

"$PYTHON_BIN" src/evaluate_predictions.py \
  --pred-path "$RUNS_ROOT/prithvi_consistency_2023/pred_${TEST_YEAR}.tif" \
  --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
  --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
  --output-dir "$RUNS_ROOT/prithvi_consistency_2023/eval_${TEST_YEAR}"

echo "Pipeline finished successfully."
