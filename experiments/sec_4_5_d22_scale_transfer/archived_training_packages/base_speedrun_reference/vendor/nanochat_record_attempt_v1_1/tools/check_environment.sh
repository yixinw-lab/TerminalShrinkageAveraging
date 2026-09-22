#!/usr/bin/env bash
set -euo pipefail
REPO=${1:-"$HOME/nanochat-record"}
EXPECTED_GPUS=${EXPECTED_GPUS:-8}
cd "$REPO"
source .venv/bin/activate

echo "host=$(hostname)"
echo "arch=$(uname -m)"
echo "commit=$(git rev-parse HEAD)"
python --version
python - <<PY
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))
assert torch.cuda.device_count() >= int(${EXPECTED_GPUS}), (torch.cuda.device_count(), ${EXPECTED_GPUS})
import liger_kernel
print('liger_kernel import PASS')
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

grep -q 'RECORD_ATTEMPT_V1' scripts/base_train.py
grep -q 'record_snapshot_manager.capture' scripts/base_train.py
grep -q 'record_terminal_lr_clamp_frac' scripts/base_train.py

echo "environment and record patch PASS"
