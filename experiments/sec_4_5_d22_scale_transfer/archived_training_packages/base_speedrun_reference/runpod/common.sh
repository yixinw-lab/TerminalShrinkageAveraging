#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$SUITE_DIR/config/experiment.env"
ROOT=${ROOT:-/workspace/nanochat-causal-speedrun-v2}
PACKAGE_DIR=${PACKAGE_DIR:-$SUITE_DIR/vendor/nanochat_record_attempt_v1_1}
REPO=${REPO:-$ROOT/nanochat-record}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$ROOT/data}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$ROOT/cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$ROOT/uv/python}
export UV_TOOL_DIR=${UV_TOOL_DIR:-$ROOT/uv/tools}
export UV_TOOL_BIN_DIR=${UV_TOOL_BIN_DIR:-$ROOT/bin}
export PATH="$ROOT/bin:$HOME/.local/bin:$PATH"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT/cache/torchinductor}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-disabled}
mkdir -p "$ROOT" "$ROOT/bin" "$ROOT/cache" "$NANOCHAT_BASE_DIR" "$ROOT/logs" "$ROOT/results" "$ROOT/snapshot_scratch"

need_8_gpus() {
  local n
  n=$(nvidia-smi -L | wc -l | tr -d ' ')
  [[ "$n" == "8" ]] || { echo "ERROR: expected 8 GPUs, found $n" >&2; return 2; }
}

activate_repo() {
  [[ -f "$ROOT/SETUP_DONE" ]] || { echo "ERROR: run bash runpod/setup_once.sh first" >&2; return 2; }
  source "$REPO/.venv/bin/activate"
  local actual
  actual=$(git -C "$REPO" rev-parse HEAD)
  [[ "$actual" == "$PR_COMMIT" ]] || { echo "ERROR: repo commit $actual != $PR_COMMIT" >&2; return 2; }
}

final_step_for_tag() {
  local tag=$1 dir="$NANOCHAT_BASE_DIR/base_checkpoints/$1"
  python - "$dir" <<'PY'
from pathlib import Path
import re,sys
p=Path(sys.argv[1])
steps=[]
for x in p.glob('meta_*.json'):
    m=re.fullmatch(r'meta_(\d+)\.json', x.name)
    if m: steps.append(int(m.group(1)))
if not steps: raise SystemExit(f'no meta_*.json under {p}')
print(max(steps))
PY
}

copy_final_meta() {
  local tag=$1 step=$2 dest=$3
  cp "$NANOCHAT_BASE_DIR/base_checkpoints/$tag/meta_$(printf '%06d' "$step").json" "$dest/"
}

run_canonical_eval() {
  local tag=$1 step=$2 ratio=$3 recipe=$4 out_json=$5 out_log=$6
  python "$PACKAGE_DIR/record_tools/eval_candidate.py" \
    --repo "$REPO" --model-tag "$tag" --step "$step" --ratio "$ratio" --recipe "$recipe" \
    --nproc 8 --split-tokens "$CANONICAL_SPLIT_TOKENS" --device-batch-size "$DEVICE_BATCH_SIZE" \
    --max-per-task -1 --output "$out_json" --log "$out_log"
}

cleanup_checkpoint_tag() {
  local tag=$1
  if [[ "${KEEP_CHECKPOINTS:-0}" != "1" ]]; then
    rm -rf "$NANOCHAT_BASE_DIR/base_checkpoints/$tag"
  fi
}
