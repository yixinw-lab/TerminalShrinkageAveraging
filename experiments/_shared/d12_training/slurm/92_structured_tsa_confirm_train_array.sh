#!/bin/bash
#SBATCH --job-name=d12sttrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${REPO:?}/slurm/_skip_env.sh"
source "${REPO}/slurm/_paper_python.sh"
ROW=$(awk -F, -v n="$((SLURM_ARRAY_TASK_ID+2))" 'NR==n {print; exit}' "${CONFIRM_MANIFEST:?}")
IFS=, read -r ROLE OPT FLOOR SEED <<< "$ROW"
[[ -n "$ROLE" && -n "$OPT" && -n "$FLOOR" && -n "$SEED" ]] || { echo "bad manifest row: $ROW" >&2; exit 2; }
case "$OPT" in
  native) PACK="$PACK_NATIVE" ;;
  pure_adamw) PACK="$PACK_ADAMW" ;;
  *) echo "unknown optimizer $OPT" >&2; exit 2 ;;
esac
case "$FLOOR" in
  0.05) FTAG=05pct ;;
  0.10) FTAG=10pct ;;
  0.15) FTAG=15pct ;;
  *) echo "unknown floor $FLOOR" >&2; exit 2 ;;
esac
OUT="${ROOT:?}/confirmation/$OPT/floor_${FTAG}/seed${SEED}"
mkdir -p "$OUT"
"$PAPER_PYTHON" -m nanochat_meta.structured_tsa_d12 train \
  --pack "$PACK" \
  --out "$OUT" \
  --group-schema "$ROOT/GROUP_SCHEMA.json" \
  --role "$ROLE" \
  --optimizer-label "$OPT" \
  --terminal-floor "$FLOOR" \
  --stream-seed "$SEED" \
  --train-steps 3000 \
  --schedule-total-iterations 3000 \
  --snapshot-dtype bfloat16 \
  --curve-eval-batches 32 \
  --alpha-select-batches 64 \
  --floor-select-batches 64 \
  --holdout-eval-batches 96 \
  --seed 1337
