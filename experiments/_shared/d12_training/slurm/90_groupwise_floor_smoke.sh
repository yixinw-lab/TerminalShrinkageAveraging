#!/bin/bash
#SBATCH --job-name=d12gwfsmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100g
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?}/slurm/_groupwise_floor_env.sh"
mkdir -p "${ROOT:?}"
"$PAPER_PYTHON" -m nanochat_meta.groupwise_floor_d12 smoke \
  --native-pack "${PACK_NATIVE:?}" \
  --protocol "${PROTOCOL:?}" \
  --out "$ROOT" \
  --train-steps 3000 \
  --schedule-total-iterations 3000 \
  --seed 1337
