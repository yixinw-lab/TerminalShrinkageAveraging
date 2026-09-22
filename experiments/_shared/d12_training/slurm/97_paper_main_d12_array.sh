#!/bin/bash
#SBATCH --job-name=papd12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=90g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

source "${REPO:?REPO must be exported}/slurm/_skip_env.sh"
source "${REPO}/slurm/_paper_python.sh"
echo "paper python: $PAPER_PYTHON ($($PAPER_PYTHON -V 2>&1))"
mapfile -t FLOORS < <(grep -vE '^\s*(#|$)' "${FLOORS_FILE:?FLOORS_FILE must be exported}")
FLOOR="${FLOORS[${SLURM_ARRAY_TASK_ID:-0}]}"
ALPHA_GRID="$(grep -vE '^\s*(#|$)' "${ALPHA_GRID_FILE:?}" | head -1)"
CURVE_STEPS="$(grep -vE '^\s*(#|$)' "${CURVE_STEPS_FILE:?}" | head -1)"
ID=$("$PAPER_PYTHON" - <<PY
f=float("$FLOOR")
print(f"floor_{100*f:05.2f}pct".replace('.', 'p'))
PY
)
OUT="${ROOT:?ROOT must be exported}/runs/${ID}"
mkdir -p "$OUT"

"$PAPER_PYTHON" -m nanochat_meta.paper_main_d12 branch \
  --pack "${PACK:?PACK must be exported}" \
  --out "$OUT" \
  --terminal-floor "$FLOOR" \
  --lr-scale 0.40 \
  --train-steps 3000 \
  --schedule-total-iterations 3000 \
  --snapshot-dtype bfloat16 \
  --alpha-grid "$ALPHA_GRID" \
  --curve-steps "$CURVE_STEPS" \
  --curve-eval-batches 32 \
  --alpha-select-batches 64 \
  --floor-select-batches 64 \
  --holdout-eval-batches 96 \
  --seed 1337
