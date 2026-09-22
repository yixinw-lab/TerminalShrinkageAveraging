#!/bin/bash
#SBATCH --job-name=d12tgsdev
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
ROW=$(awk -F, -v n="$((SLURM_ARRAY_TASK_ID+2))" 'NR==n {print; exit}' "${DEV_MANIFEST:?}" | tr -d '\r')
IFS=, read -r OPT FLOOR SEED <<< "$ROW"
[[ -n "$OPT" && -n "$FLOOR" && -n "$SEED" ]] || { echo "bad dev manifest row: $ROW" >&2; exit 2; }
[[ "$OPT" == "native" ]] || { echo "development is native-only, got $OPT" >&2; exit 2; }
OUT="${ROOT:?}/dev/$OPT/floor_10pct/seed${SEED}"
mkdir -p "$OUT"
[[ ! -f "$OUT/result.json" ]] || { echo "result already exists: $OUT/result.json" >&2; exit 2; }
"$PAPER_PYTHON" -m nanochat_meta.tsa_group_search_d12 develop \
  --pack "${PACK_NATIVE:?}" \
  --out "$OUT" \
  --optimizer-label "$OPT" \
  --terminal-floor "$FLOOR" \
  --stream-seed "$SEED" \
  --train-steps 3000 \
  --schedule-total-iterations 3000 \
  --search-eval-batches 32 \
  --holdout-eval-batches 96 \
  --seed 1337
