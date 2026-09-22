#!/bin/bash
#SBATCH --job-name=d12stsmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110g
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_skip_env.sh"
source "${REPO}/slurm/_paper_python.sh"
"$PAPER_PYTHON" -m nanochat_meta.structured_tsa_d12 smoke \
  --native-pack "${PACK_NATIVE:?}" \
  --adamw-pack "${PACK_ADAMW:?}" \
  --out "${ROOT:?}"
