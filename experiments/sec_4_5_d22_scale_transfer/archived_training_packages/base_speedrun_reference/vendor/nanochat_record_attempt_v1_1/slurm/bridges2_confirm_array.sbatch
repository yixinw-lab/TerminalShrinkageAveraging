#!/usr/bin/env bash
#SBATCH --job-name=nc_record_confirm
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h100-80:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=logs/nc_record_confirm_%A_%a.out
#SBATCH --error=logs/nc_record_confirm_%A_%a.err
set -euo pipefail
: "${REPO:?}"
: "${NANOCHAT_BASE_DIR:?}"
: "${RECORD_PACKAGE_DIR:?}"
: "${FROZEN_ENV:?}"
source "$FROZEN_ENV"

cd "$REPO"
[[ -z "${BRIDGES2_MODULES:-}" ]] || module load ${BRIDGES2_MODULES}
source .venv/bin/activate
export NANOCHAT_BASE_DIR OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

TASK=${SLURM_ARRAY_TASK_ID:-0}
RUN_TAG="record_confirm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}"
RESULT_DIR="$NANOCHAT_BASE_DIR/record_attempts/confirm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"
LOCAL_ROOT=${RECORD_LOCAL_ROOT:-${LOCAL:-${SLURM_TMPDIR:-/tmp/$USER}}}
SNAP_DIR="$LOCAL_ROOT/nanochat_record_confirm_${SLURM_JOB_ID}_${TASK}"
mkdir -p "$SNAP_DIR"
TRAIN_TAG="${RUN_TAG}_train"

set -x
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=22 \
  --target-param-data-ratio="$TARGET_RATIO" \
  --record-schedule-param-data-ratio="$SCHEDULE_RATIO" \
  --record-terminal-lr-clamp-frac="$TERMINAL_CLAMP_FRAC" \
  --record-snapshot-ratios="$TARGET_RATIO" \
  --record-snapshot-k="$SNAPSHOT_K" \
  --record-snapshot-spacing-frac="$SNAPSHOT_SPACING_FRAC" \
  --record-snapshot-dir="$SNAP_DIR" \
  --device-batch-size=32 --total-batch-size=524288 \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=999999 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --model-tag="$TRAIN_TAG" --run="${WANDB_PREFIX:-record_confirm}_${TASK}"
set +x

python "$RECORD_PACKAGE_DIR/record_tools/merge_record_snapshots.py" \
  --snapshot-dir "$SNAP_DIR" \
  --source-checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/$TRAIN_TAG" \
  --base-checkpoints "$NANOCHAT_BASE_DIR/base_checkpoints" \
  --output-prefix "$RUN_TAG" \
  --recipes "$RECIPE" \
  --output-manifest "$RESULT_DIR/candidates_${TASK}.csv"

mapfile -t FIELDS < <(python - "$RESULT_DIR/candidates_${TASK}.csv" <<'PY'
import csv,sys
r=next(csv.DictReader(open(sys.argv[1])))
for k in ('model_tag','endpoint_step','ratio','recipe'):
    print(r[k])
PY
)
MODEL_TAG=${FIELDS[0]}; STEP=${FIELDS[1]}; RATIO=${FIELDS[2]}; RECIPE_USED=${FIELDS[3]}
python "$RECORD_PACKAGE_DIR/record_tools/eval_candidate.py" \
  --repo "$REPO" --model-tag "$MODEL_TAG" --step "$STEP" \
  --ratio "$RATIO" --recipe "$RECIPE_USED" --nproc 8 \
  --split-tokens "${CANONICAL_SPLIT_TOKENS:-20971520}" --device-batch-size 32 --max-per-task -1 \
  --output "$RESULT_DIR/confirm_${TASK}.json" \
  --log "$RESULT_DIR/confirm_${TASK}.log"
cp "$SNAP_DIR/manifest.json" "$RESULT_DIR/snapshot_manifest_${TASK}.json"
