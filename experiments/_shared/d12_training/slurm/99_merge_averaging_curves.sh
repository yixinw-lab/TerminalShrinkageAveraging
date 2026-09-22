#!/bin/bash
#SBATCH --job-name=avgcurve_merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source "${REPO:?REPO must be exported}/slurm/_skip_env.sh"
GROUP_ROOT="${GROUP_ROOT:?GROUP_ROOT must be exported}"
OUT="${GROUP_ROOT}/figures"
python -m nanochat_meta.averaging_curve_plot --root "$GROUP_ROOT" --out "$OUT"
cat "$OUT/DIGEST.txt"
