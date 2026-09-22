#!/bin/bash
#SBATCH --job-name=d12gwfrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${REPO:?}/slurm/_groupwise_floor_env.sh"
ROW=$(awk -F, -v n="$((SLURM_ARRAY_TASK_ID+2))" 'NR==n {print; exit}' "${RUN_MANIFEST:?}" | tr -d '\r')
IFS=, read -r ARM SEED <<< "$ROW"
[[ -n "$ARM" && -n "$SEED" ]] || { echo "bad manifest row: $ROW" >&2; exit 2; }
case "$ARM" in
  uniform05|uniform10|matched_uniform|mapped) ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
OUT="${ROOT:?}/runs/$ARM/seed${SEED}"
mkdir -p "$OUT"
[[ ! -f "$OUT/result.json" ]] || { echo "result already exists: $OUT/result.json" >&2; exit 2; }
"$PAPER_PYTHON" -m nanochat_meta.groupwise_floor_d12 run \
  --pack "${PACK_NATIVE:?}" \
  --protocol "${PROTOCOL:?}" \
  --out "$OUT" \
  --arm "$ARM" \
  --stream-seed "$SEED" \
  --train-steps 3000 \
  --schedule-total-iterations 3000 \
  --holdout-eval-batches 96 \
  --seed 1337
