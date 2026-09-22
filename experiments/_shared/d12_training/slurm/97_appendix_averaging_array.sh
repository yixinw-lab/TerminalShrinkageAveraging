#!/bin/bash
#SBATCH --job-name=d12avgapp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${REPO:?}/slurm/_skip_env.sh"
source "${REPO}/slurm/_paper_python.sh"
SEED=$(grep -vE '^\s*(#|$)' "${SEEDS_FILE:?}" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
[[ -n "$SEED" ]] || { echo 'missing seed' >&2; exit 2; }
OUT="${ROOT:?}/runs/seed${SEED}"
mkdir -p "$OUT"
"$PAPER_PYTHON" -m nanochat_meta.appendix_averaging_d12 run \
  --pack "${PACK:?}" --out "$OUT" \
  --terminal-floor 0.10 --alpha 0.55 \
  --train-steps 3000 --schedule-total-iterations 3000 \
  --snapshot-dtype bfloat16 \
  --curve-eval-batches 32 --alpha-select-batches 64 --floor-select-batches 64 --holdout-eval-batches 96 \
  --seed 1337 --stream-seed "$SEED"
