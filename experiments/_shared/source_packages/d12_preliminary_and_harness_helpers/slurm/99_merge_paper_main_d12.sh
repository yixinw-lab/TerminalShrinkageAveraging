#!/bin/bash
#SBATCH --job-name=papd12merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?REPO must be exported}/slurm/_skip_env.sh"
source "${REPO}/slurm/_paper_python.sh"
echo "paper python: $PAPER_PYTHON ($($PAPER_PYTHON -V 2>&1))"
"$PAPER_PYTHON" -m nanochat_meta.paper_main_d12 merge \
  --root "${ROOT:?ROOT must be exported}" \
  --floors "0.05,0.10,0.125,0.15,0.175"
