#!/bin/bash
#SBATCH --job-name=d12stmerge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_skip_env.sh"
source "${REPO}/slurm/_paper_python.sh"
"$PAPER_PYTHON" "${REPO}/tools/analyze_structured_tsa_d12.py" --root "${ROOT:?}"
