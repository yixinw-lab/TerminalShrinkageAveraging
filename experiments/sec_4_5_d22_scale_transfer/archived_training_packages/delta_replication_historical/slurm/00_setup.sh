#!/usr/bin/env bash
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --job-name=d22dsetup
#SBATCH --output=d22dsetup_%j.log
set -euo pipefail
SUITE_DIR="${SUITE_DIR:?set SUITE_DIR}"
source "$SUITE_DIR/slurm/_d22_env.sh"
# Prefer the exact NanoChat lock/environment. The site module is only a bootstrap for uv.
if [[ ! -f "$NANOCHAT_SRC/.venv/bin/activate" ]]; then
  module load python/miniforge3_pytorch/2.10.0
  command -v uv >/dev/null 2>&1 || python -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
  uv sync --extra gpu --extra liger
  source .venv/bin/activate
fi
python - <<'PY'
import torch
print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())
assert torch.cuda.is_available()
print('gpu',torch.cuda.get_device_name(0))
PY
# Exact PR830 data/tokenizer preparation, isolated from other experiments.
mkdir -p "$NANOCHAT_BASE_DIR"
python -m nanochat.dataset -n 8
python -m scripts.tok_train --vocab-size=49152
python -m scripts.tok_eval
python -m nanochat.dataset -n 170
python - <<'PY'
from nanochat.tokenizer import get_tokenizer
v=get_tokenizer().get_vocab_size(); print('vocab',v); assert v==49152
PY
python -m py_compile scripts/base_train_d22_delta.py
mkdir -p "$RESULT_ROOT"
{
  echo "setup_complete=$(date -Is)"
  echo "host=$(hostname)"
  echo "python=$(which python)"
  python - <<'PY'
import torch,sys
print('python_version='+sys.version.replace('\n',' '))
print('torch='+torch.__version__)
print('cuda='+str(torch.version.cuda))
print('gpu='+torch.cuda.get_device_name(0))
PY
} > "$RESULT_ROOT/SETUP.txt"
echo 'SETUP PASS'
