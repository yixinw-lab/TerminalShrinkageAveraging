#!/bin/bash
#SBATCH --job-name=d12tgsfreeze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_tsa_group_search_env.sh"
"$PAPER_PYTHON" "${REPO}/tools/analyze_tsa_group_search_d12.py" \
  --root "${ROOT:?}" \
  --freeze \
  --expected-dev 8
