#!/usr/bin/env bash
set -euo pipefail
export WORK_ROOT="${WORK_ROOT:-/work/nvme/bhji/$USER/ExtrapProj/d22_delta_replication_v1}"
export NANOCHAT_SRC="${NANOCHAT_SRC:-$WORK_ROOT/nanochat}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$WORK_ROOT/nanochat_base}"
export RESULT_ROOT="${RESULT_ROOT:-$WORK_ROOT/results}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
mkdir -p "$NANOCHAT_BASE_DIR" "$RESULT_ROOT/logs" "$RESULT_ROOT/runs"
cd "$NANOCHAT_SRC"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  module load python/miniforge3_pytorch/2.10.0
  conda activate base
fi
