#!/bin/bash
#SBATCH --job-name=tree-canopy-full
#SBATCH --partition=a30_normal_q
#SBATCH --account=cs5814
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dl}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_DIR/runs/$RUN_TAG}"
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
TRAIN_STRIDE="${TRAIN_STRIDE:-64}"
EVAL_STRIDE="${EVAL_STRIDE:-64}"
INFER_STRIDE="${INFER_STRIDE:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

RF_SAMPLE_FRACTION="${RF_SAMPLE_FRACTION:-0.05}"
PRITHVI_100M_CKPT="${PRITHVI_100M_CKPT:-$PROJECT_DIR/model_prithivi_weights/Prithvi_EO_V2_100M_TL.pt}"
PRITHVI_300M_CKPT="${PRITHVI_300M_CKPT:-$PROJECT_DIR/model_prithivi_weights/Prithvi_EO_V2_300M.pt}"

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
echo "Run tag: $RUN_TAG"
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

infer_prithvi_variant_tag() {
  local ckpt_name
  ckpt_name="$(basename "$1")"
  ckpt_name="${ckpt_name,,}"
  if [[ "$ckpt_name" == *"300m"* ]]; then
    echo "prithvi_300m"
  elif [[ "$ckpt_name" == *"100m"* ]]; then
    echo "prithvi_100m"
  elif [[ "$ckpt_name" == *"tiny"* ]]; then
    echo "prithvi_tiny"
  else
    echo "prithvi"
  fi
}

TRAIN_YEARS_STR="$(join_by_space "${TRAIN_YEARS[@]}")"
SUPERVISED_YEARS_STR="$(join_by_space "${SUPERVISED_YEARS[@]}")"
CONSISTENCY_YEARS_STR="$(join_by_space "${CONSISTENCY_YEARS[@]}")"

run_prithvi_variant() {
  local prithvi_ckpt="$1"
  local prithvi_variant_tag
  local prithvi_base_dir
  local prithvi_consistency_dir

  prithvi_variant_tag="$(infer_prithvi_variant_tag "$prithvi_ckpt")"
  prithvi_base_dir="$RUNS_ROOT/${prithvi_variant_tag}_2023"
  prithvi_consistency_dir="$RUNS_ROOT/${prithvi_variant_tag}_consistency_2023"

  echo "== ${prithvi_variant_tag} baseline =="
  "$PYTHON_BIN" src/train_deep.py \
    --data-root "$DATA_ROOT" \
    --train-years $TRAIN_YEARS_STR \
    --supervised-years $SUPERVISED_YEARS_STR \
    --consistency-years $CONSISTENCY_YEARS_STR \
    --test-year "$TEST_YEAR" \
    --reference-year "$REFERENCE_YEAR" \
    --output-dir "$prithvi_base_dir" \
    --model geo_vit \
    --pretrained-checkpoint "$prithvi_ckpt" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --stride "$STRIDE" \
    --train-stride "$TRAIN_STRIDE" \
    --eval-stride "$EVAL_STRIDE" \
    --block-size "$BLOCK_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --consistency-weight 0.0

  "$PYTHON_BIN" src/predict_raster.py \
    --data-root "$DATA_ROOT" \
    --year "$TEST_YEAR" \
    --checkpoint "$prithvi_base_dir/best.pt" \
    --stats-path "$prithvi_base_dir/stats.json" \
    --output-path "$prithvi_base_dir/pred_${TEST_YEAR}.tif" \
    --stride "$INFER_STRIDE" \
    --tta

  "$PYTHON_BIN" src/evaluate_predictions.py \
    --pred-path "$prithvi_base_dir/pred_${TEST_YEAR}.tif" \
    --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
    --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
    --output-dir "$prithvi_base_dir/eval_${TEST_YEAR}"

  echo "== ${prithvi_variant_tag} + consistency =="
  "$PYTHON_BIN" src/train_deep.py \
    --data-root "$DATA_ROOT" \
    --train-years $TRAIN_YEARS_STR \
    --supervised-years $SUPERVISED_YEARS_STR \
    --consistency-years $CONSISTENCY_YEARS_STR \
    --test-year "$TEST_YEAR" \
    --reference-year "$REFERENCE_YEAR" \
    --output-dir "$prithvi_consistency_dir" \
    --model geo_vit \
    --pretrained-checkpoint "$prithvi_ckpt" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --stride "$STRIDE" \
    --train-stride "$TRAIN_STRIDE" \
    --eval-stride "$EVAL_STRIDE" \
    --block-size "$BLOCK_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --consistency-weight 0.2 \
    --change-gamma 3.0

  "$PYTHON_BIN" src/predict_raster.py \
    --data-root "$DATA_ROOT" \
    --year "$TEST_YEAR" \
    --checkpoint "$prithvi_consistency_dir/best.pt" \
    --stats-path "$prithvi_consistency_dir/stats.json" \
    --output-path "$prithvi_consistency_dir/pred_${TEST_YEAR}.tif" \
    --stride "$INFER_STRIDE" \
    --tta

  "$PYTHON_BIN" src/evaluate_predictions.py \
    --pred-path "$prithvi_consistency_dir/pred_${TEST_YEAR}.tif" \
    --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
    --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
    --output-dir "$prithvi_consistency_dir/eval_${TEST_YEAR}"
}

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
  --train-stride "$TRAIN_STRIDE" \
  --eval-stride "$EVAL_STRIDE" \
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
  --stride "$INFER_STRIDE" \
  --tta

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
  --train-stride "$TRAIN_STRIDE" \
  --eval-stride "$EVAL_STRIDE" \
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
  --stride "$INFER_STRIDE" \
  --tta

"$PYTHON_BIN" src/evaluate_predictions.py \
  --pred-path "$RUNS_ROOT/unet_consistency_2023/pred_${TEST_YEAR}.tif" \
  --target-path "$DATA_ROOT/$TEST_YEAR/Tree_${TEST_YEAR}.tif" \
  --reference-path "$DATA_ROOT/$TEST_YEAR/${TEST_YEAR}_Tree_NLCD.tif" \
  --output-dir "$RUNS_ROOT/unet_consistency_2023/eval_${TEST_YEAR}"

run_prithvi_variant "$PRITHVI_100M_CKPT"
run_prithvi_variant "$PRITHVI_300M_CKPT"

echo "Pipeline finished successfully."
