#!/bin/bash
#SBATCH --job-name=d12tgsconf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${REPO:?}/slurm/_tsa_group_search_env.sh"
ROW=$(awk -F, -v n="$((SLURM_ARRAY_TASK_ID+2))" 'NR==n {print; exit}' "${CONFIRM_MANIFEST:?}" | tr -d '\r')
IFS=, read -r OPT FLOOR SEED <<< "$ROW"
[[ -n "$OPT" && -n "$FLOOR" && -n "$SEED" ]] || { echo "bad confirm manifest row: $ROW" >&2; exit 2; }
case "$OPT" in
  native) PACK="$PACK_NATIVE" ;;
  pure_adamw) PACK="$PACK_ADAMW" ;;
  *) echo "unknown optimizer $OPT" >&2; exit 2 ;;
esac
OUT="${ROOT:?}/confirm/$OPT/floor_10pct/seed${SEED}"
mkdir -p "$OUT"
[[ ! -f "$OUT/result.json" ]] || { echo "result already exists: $OUT/result.json" >&2; exit 2; }
"$PAPER_PYTHON" -m nanochat_meta.tsa_group_search_d12 confirm \
  --pack "$PACK" \
  --frozen-rules "${FROZEN_RULES:?}" \
  --out "$OUT" \
  --optimizer-label "$OPT" \
  --terminal-floor "$FLOOR" \
  --stream-seed "$SEED" \
  --train-steps 3000 \
  --schedule-total-iterations 3000 \
  --holdout-eval-batches 96 \
  --seed 1337
