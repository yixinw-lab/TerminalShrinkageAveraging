#!/usr/bin/env bash
#SBATCH --job-name=nc_record_probe
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=logs/nc_record_probe_%j.out
#SBATCH --error=logs/nc_record_probe_%j.err
set -euo pipefail
# Functional/transfer probe only: 4-GPU GH200 is not official 8xH100 leaderboard hardware.
: "${REPO:?}"; : "${NANOCHAT_BASE_DIR:?}"; : "${RECORD_PACKAGE_DIR:?}"; : "${RUN_TAG:?}"
cd "$REPO"
source .venv/bin/activate
export NANOCHAT_BASE_DIR OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
EXPECTED_GPUS=4 "$RECORD_PACKAGE_DIR/tools/check_environment.sh" "$REPO"
LOCAL_ROOT=${RECORD_LOCAL_ROOT:-${SLURM_TMPDIR:-/tmp/$USER}}
SNAP_DIR="$LOCAL_ROOT/nanochat_record_probe_${SLURM_JOB_ID}"
RESULT_DIR="$NANOCHAT_BASE_DIR/record_attempts/$RUN_TAG"
mkdir -p "$SNAP_DIR" "$RESULT_DIR"
TRAIN_TAG="${RUN_TAG}_train"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
  --depth=22 --target-param-data-ratio="${TRAIN_RATIO:-9.4}" \
  --record-schedule-param-data-ratio="${SCHEDULE_RATIO:-9.4}" \
  --record-terminal-lr-clamp-frac="${TERMINAL_CLAMP_FRAC:-0.15}" \
  --record-snapshot-ratios="${SNAPSHOT_RATIOS:-9.2,9.4}" \
  --record-snapshot-k="${SNAPSHOT_K:-8}" \
  --record-snapshot-spacing-frac="${SNAPSHOT_SPACING_FRAC:-0.010666666666666666}" \
  --record-snapshot-dir="$SNAP_DIR" \
  --device-batch-size=32 --total-batch-size=524288 \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=999999 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --model-tag="$TRAIN_TAG" --run="${WANDB_RUN:-dummy}"
python "$RECORD_PACKAGE_DIR/record_tools/merge_record_snapshots.py" \
  --snapshot-dir "$SNAP_DIR" --source-checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/$TRAIN_TAG" \
  --base-checkpoints "$NANOCHAT_BASE_DIR/base_checkpoints" --output-prefix "$RUN_TAG" \
  --recipes-file "${RECIPES_FILE:-$RECORD_PACKAGE_DIR/config/recipes_compact.txt}" \
  --output-manifest "$RESULT_DIR/candidates.csv"
python "$RECORD_PACKAGE_DIR/record_tools/screen_candidates.py" \
  --repo "$REPO" --candidates "$RESULT_DIR/candidates.csv" --output-dir "$RESULT_DIR/bpb_screen" \
  --nproc 4 --split-tokens "${SCREEN_SPLIT_TOKENS:-2097152}" --device-batch-size 32
cp "$SNAP_DIR/manifest.json" "$RESULT_DIR/snapshot_manifest.json"
