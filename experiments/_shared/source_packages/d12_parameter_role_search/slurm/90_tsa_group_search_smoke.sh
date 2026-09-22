#!/bin/bash
#SBATCH --job-name=d12tgssmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100g
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_tsa_group_search_env.sh"
mkdir -p "${ROOT:?}"
"$PAPER_PYTHON" -m nanochat_meta.tsa_group_search_d12 smoke \
  --native-pack "${PACK_NATIVE:?}" \
  --out "$ROOT" \
  --seed 1337
