#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
need_8_gpus
exec > >(tee -a "$ROOT/logs/setup_once.log") 2>&1

echo "===== ONE-TIME RUNPOD SETUP ====="
date -u +%FT%TZ
nvidia-smi -L
nvidia-smi topo -m

df -h /workspace
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh
  hash -r
fi
uv --version

if [[ ! -d "$REPO/.git" ]]; then
  bash "$PACKAGE_DIR/tools/setup_pr830.sh" "$REPO"
else
  source "$REPO/.venv/bin/activate"
  actual=$(git -C "$REPO" rev-parse HEAD)
  [[ "$actual" == "$PR_COMMIT" ]] || { echo "ERROR: existing repo at $actual" >&2; exit 2; }
  grep -q 'RECORD_ATTEMPT_V1' "$REPO/scripts/base_train.py" || bash "$PACKAGE_DIR/tools/finish_existing_checkout.sh" "$REPO"
fi
source "$REPO/.venv/bin/activate"

if [[ ! -s "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]] || [[ $(find "$NANOCHAT_BASE_DIR/base_data_climbmix" -type f 2>/dev/null | wc -l) -lt 170 ]]; then
  bash "$PACKAGE_DIR/tools/prepare_data.sh" "$REPO"
fi
EXPECTED_GPUS=8 "$PACKAGE_DIR/tools/check_environment.sh" "$REPO"
FREE_GIB=$(df -Pk /workspace | awk 'NR==2 {printf "%d", $4/1024/1024}')
echo "free_workspace_gib=$FREE_GIB"
(( FREE_GIB >= 90 )) || { echo "ERROR: require >=90 GiB free after prep" >&2; exit 3; }
touch "$ROOT/SETUP_DONE"
echo "SETUP COMPLETE"
