#!/usr/bin/env bash
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=d22dmerge
#SBATCH --output=d22dmerge_%j.log
set -euo pipefail
SUITE_DIR="${SUITE_DIR:?set SUITE_DIR}"
source "$SUITE_DIR/slurm/_d22_env.sh"
python "$SUITE_DIR/tools/analyze_d22_delta.py" "$RESULT_ROOT"
