#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${1:-"$HOME/nanochat-record"}
PR_COMMIT=e09bc164162f35da0b5b8315be791e9e974a4c3a

if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/karpathy/nanochat.git "$DEST"
fi
cd "$DEST"
if ! git fetch origin pull/830/head:refs/remotes/origin/pr830; then
  git remote remove pr830-fork >/dev/null 2>&1 || true
  git remote add pr830-fork https://github.com/giovannizinzi/nanochat-gio.git
  git fetch pr830-fork gio/d22-liger-rmsnorm-speedrun
fi
if ! git cat-file -e "$PR_COMMIT^{commit}" 2>/dev/null; then
  git remote get-url pr830-fork >/dev/null 2>&1 || git remote add pr830-fork https://github.com/giovannizinzi/nanochat-gio.git
  git fetch pr830-fork gio/d22-liger-rmsnorm-speedrun
fi
git checkout -B record-pr830 "$PR_COMMIT"
ACTUAL=$(git rev-parse HEAD)
[[ "$ACTUAL" == "$PR_COMMIT" ]] || { echo "commit mismatch: $ACTUAL" >&2; exit 1; }

cp "$PACKAGE_DIR/nanochat/record_snapshots.py" nanochat/record_snapshots.py

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/ and rerun." >&2
  exit 1
fi
uv sync --extra gpu --extra liger
source .venv/bin/activate
python "$PACKAGE_DIR/tools/apply_record_patch.py" "$DEST"
python -m py_compile scripts/base_train.py scripts/base_eval.py nanochat/record_snapshots.py
python -m nanochat.record_snapshots
python "$PACKAGE_DIR/record_tools/merge_record_snapshots.py" --selftest

cat > RECORD_ATTEMPT_PROVENANCE.json <<EOF
{
  "upstream": "https://github.com/karpathy/nanochat",
  "pull_request": 830,
  "commit": "$PR_COMMIT",
  "record_package": "nanochat_record_attempt_v1"
}
EOF

echo
echo "record repo ready: $DEST"
echo "commit: $ACTUAL"
echo "next: export NANOCHAT_BASE_DIR=<persistent path>; bash $PACKAGE_DIR/tools/prepare_data.sh $DEST"
