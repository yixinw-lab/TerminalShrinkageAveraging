#!/usr/bin/env bash
#SBATCH --job-name=nc_record_explore
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h100-80:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/nc_record_explore_%j.out
#SBATCH --error=logs/nc_record_explore_%j.err
set -euo pipefail

: "${REPO:?}"
: "${NANOCHAT_BASE_DIR:?}"
: "${RECORD_PACKAGE_DIR:?}"
: "${RUN_TAG:?}"
: "${TRAIN_RATIO:=9.4}"
: "${SCHEDULE_RATIO:=9.4}"
: "${SNAPSHOT_RATIOS:=9.0,9.1,9.2,9.3,9.4}"
: "${TERMINAL_CLAMP_FRAC:=0.15}"
: "${SNAPSHOT_K:=8}"
: "${SNAPSHOT_SPACING_FRAC:=0.010666666666666666}"
: "${RECIPES_FILE:=$RECORD_PACKAGE_DIR/config/recipes_compact.txt}"
: "${SCREEN_SPLIT_TOKENS:=4194304}"
: "${WANDB_RUN:=dummy}"

mkdir -p "$RECORD_PACKAGE_DIR/logs" "$NANOCHAT_BASE_DIR/record_attempts/$RUN_TAG"
cd "$REPO"
[[ -z "${BRIDGES2_MODULES:-}" ]] || module load ${BRIDGES2_MODULES}
source .venv/bin/activate
export NANOCHAT_BASE_DIR
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

EXPECTED_GPUS=8 "$RECORD_PACKAGE_DIR/tools/check_environment.sh" "$REPO"
LOCAL_ROOT=${RECORD_LOCAL_ROOT:-${LOCAL:-${SLURM_TMPDIR:-/tmp/$USER}}}
SNAP_DIR="$LOCAL_ROOT/nanochat_record_${SLURM_JOB_ID}"
mkdir -p "$SNAP_DIR"
RESULT_DIR="$NANOCHAT_BASE_DIR/record_attempts/$RUN_TAG"
TRAIN_TAG="${RUN_TAG}_train"

set -x
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=22 \
  --target-param-data-ratio="$TRAIN_RATIO" \
  --record-schedule-param-data-ratio="$SCHEDULE_RATIO" \
  --record-terminal-lr-clamp-frac="$TERMINAL_CLAMP_FRAC" \
  --record-snapshot-ratios="$SNAPSHOT_RATIOS" \
  --record-snapshot-k="$SNAPSHOT_K" \
  --record-snapshot-spacing-frac="$SNAPSHOT_SPACING_FRAC" \
  --record-snapshot-dir="$SNAP_DIR" \
  --device-batch-size=32 --total-batch-size=524288 \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=999999 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --model-tag="$TRAIN_TAG" --run="$WANDB_RUN"
set +x

cp "$SNAP_DIR/manifest.json" "$RESULT_DIR/snapshot_manifest.json"
python "$RECORD_PACKAGE_DIR/record_tools/merge_record_snapshots.py" \
  --snapshot-dir "$SNAP_DIR" \
  --source-checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/$TRAIN_TAG" \
  --base-checkpoints "$NANOCHAT_BASE_DIR/base_checkpoints" \
  --output-prefix "$RUN_TAG" \
  --recipes-file "$RECIPES_FILE" \
  --output-manifest "$RESULT_DIR/candidates.csv"

python "$RECORD_PACKAGE_DIR/record_tools/screen_candidates.py" \
  --repo "$REPO" \
  --candidates "$RESULT_DIR/candidates.csv" \
  --output-dir "$RESULT_DIR/bpb_screen" \
  --nproc 8 \
  --split-tokens "$SCREEN_SPLIT_TOKENS" \
  --device-batch-size 32

touch "$RESULT_DIR/EXPLORE_DONE"
echo "Explore complete: $RESULT_DIR"
echo "Next: $RECORD_PACKAGE_DIR/slurm/submit_bridges2_core.sh $RESULT_DIR/bpb_screen/top_candidates.csv"
