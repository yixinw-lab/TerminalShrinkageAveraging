#!/bin/bash
#SBATCH --job-name=fprobe_arm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${REPO:?}/slurm/_skip_env.sh"
mapfile -t ARMS < <(grep -vE '^\s*(#|$)' "${ARMS_FILE:?}")
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}";SLUG=$(printf '%s' "$ARM"|sed -E 's/[^A-Za-z0-9_.-]+/_/g');OUT="${ROOT}/arms/${SLUG}";mkdir -p "$OUT"
python -m nanochat_meta.forward_probe_fastforward branch \
 --pack "$PACK" --arm "$ARM" --out "$OUT" \
 --compute-budget "${COMPUTE_BUDGET:-1800}" --base-lr-scale "${BASE_LR_SCALE:-0.30}" --jump-start-equiv "${JUMP_START_EQUIV:-100}" --jump-stop-equiv "${JUMP_STOP_EQUIV:-1200}" \
 --eval-every-equiv "${EVAL_EVERY_EQUIV:-100}" --lawa-window "${LAWA_WINDOW:-225}" \
 --warmdown-ratio "${BRANCH_WARMDOWN_RATIO:-0}" --final-lr-frac "${BRANCH_FINAL_LR_FRAC:-0.05}" --seed "${SEED:-1337}"
