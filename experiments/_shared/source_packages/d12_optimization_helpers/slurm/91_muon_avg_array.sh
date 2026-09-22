#!/bin/bash
#SBATCH --job-name=muavg_traj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${REPO:?REPO must be exported}/slurm/_skip_env.sh"
mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "${TRAJECTORIES_FILE:?}")
ROW="${ROWS[${SLURM_ARRAY_TASK_ID:-0}]}"
IFS=',' read -r TRAJ LR WARM FINAL <<<"${ROW}"
OUT="${ROOT}/trajectories/${TRAJ}"
mkdir -p "${OUT}"
python -m nanochat_meta.muon_aware_averaging branch \
  --pack "${PACK}" --out "${OUT}" --trajectory-id "${TRAJ}" \
  --recipes-file "${RECIPES_FILE}" \
  --compute-budget "${COMPUTE_BUDGET:-1800}" \
  --lr-scale "${LR}" --warmdown-ratio "${WARM}" --final-lr-frac "${FINAL}" \
  --snapshot-every "${SNAPSHOT_EVERY:-4}" --snapshot-dtype "${SNAPSHOT_DTYPE:-bfloat16}" \
  --recipe-eval-equivs "${RECIPE_EVAL_EQUIVS:-1200,1500,1800}" \
  --eval-every-equiv "${EVAL_EVERY_EQUIV:-100}" \
  --seed "${SEED:-1337}"
