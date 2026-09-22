#!/bin/bash
#SBATCH --job-name=avgcurve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

source "${REPO:?REPO must be exported}/slurm/_skip_env.sh"
mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "${TRAJECTORIES_FILE:?TRAJECTORIES_FILE must be exported}")
ROW="${ROWS[${SLURM_ARRAY_TASK_ID:-0}]}"
IFS=',' read -r TRAJ LR WARM FINAL <<<"${ROW}"
OUT="${ROOT:?ROOT must be exported}/trajectories/${TRAJ}"
mkdir -p "${OUT}"

python -m nanochat_meta.muon_aware_averaging branch \
  --pack "${PACK:?PACK must be exported}" \
  --out "${OUT}" \
  --trajectory-id "${TRAJ}" \
  --recipes-file "${RECIPES_FILE:?RECIPES_FILE must be exported}" \
  --compute-budget "${COMPUTE_BUDGET:-3000}" \
  --lr-scale "${LR}" \
  --warmdown-ratio "${WARM}" \
  --final-lr-frac "${FINAL}" \
  --snapshot-every "${SNAPSHOT_EVERY:-4}" \
  --snapshot-dtype "${SNAPSHOT_DTYPE:-bfloat16}" \
  --recipe-eval-equivs "${RECIPE_EVAL_EQUIVS:-600,900,1200,1500,1800,2100,2400,2700,3000}" \
  --eval-every-equiv "${EVAL_EVERY_EQUIV:-300}" \
  --curve-eval-batches "${CURVE_EVAL_BATCHES:-32}" \
  --final-eval-batches "${FINAL_EVAL_BATCHES:-256}" \
  --seed "${SEED:-1337}"
