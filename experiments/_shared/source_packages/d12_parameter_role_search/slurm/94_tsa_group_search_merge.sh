#!/bin/bash
#SBATCH --job-name=d12tgsmerge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_tsa_group_search_env.sh"
EXPECTED_DEV=$(( $(wc -l < "${DEV_MANIFEST:?}") - 1 ))
EXPECTED_CONFIRM=$(( $(wc -l < "${CONFIRM_MANIFEST:?}") - 1 ))
FOUND_DEV=$(find "${ROOT:?}/dev" -name result.json -type f 2>/dev/null | wc -l)
FOUND_CONFIRM=$(find "${ROOT:?}/confirm" -name result.json -type f 2>/dev/null | wc -l)
[[ "$FOUND_DEV" -eq "$EXPECTED_DEV" ]] || { echo "expected $EXPECTED_DEV dev results, found $FOUND_DEV" >&2; exit 2; }
[[ "$FOUND_CONFIRM" -eq "$EXPECTED_CONFIRM" ]] || { echo "expected $EXPECTED_CONFIRM confirm results, found $FOUND_CONFIRM" >&2; exit 2; }
"$PAPER_PYTHON" "${REPO}/tools/analyze_tsa_group_search_d12.py" \
  --root "$ROOT" \
  --expected-confirm "$EXPECTED_CONFIRM"
