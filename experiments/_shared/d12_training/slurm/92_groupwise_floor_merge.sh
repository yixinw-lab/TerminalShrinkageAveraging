#!/bin/bash
#SBATCH --job-name=d12gwfmerge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_groupwise_floor_env.sh"
EXPECTED=$(( $(wc -l < "${RUN_MANIFEST:?}") - 1 ))
FOUND=$(find "${ROOT:?}/runs" -name result.json -type f 2>/dev/null | wc -l)
[[ "$FOUND" -eq "$EXPECTED" ]] || { echo "expected $EXPECTED results, found $FOUND" >&2; exit 2; }
"$PAPER_PYTHON" "${REPO}/tools/analyze_groupwise_floor_d12.py" "$ROOT"
