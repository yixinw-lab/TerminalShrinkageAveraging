#!/bin/bash
#SBATCH --job-name=fprobe_merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_skip_env.sh"
python -m nanochat_meta.forward_probe_plot --root "${ROOT}/arms" --out "$ROOT"
