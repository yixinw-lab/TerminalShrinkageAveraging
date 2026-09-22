#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=${1:?usage: finish_existing_checkout.sh /path/to/nanochat-record}
PR_COMMIT=e09bc164162f35da0b5b8315be791e9e974a4c3a
cd "$REPO"
ACTUAL=$(git rev-parse HEAD)
[[ "$ACTUAL" == "$PR_COMMIT" ]] || { echo "commit mismatch: $ACTUAL" >&2; exit 1; }
[[ -d .venv ]] || { echo "missing $REPO/.venv; run setup_pr830.sh first" >&2; exit 1; }
cp "$PACKAGE_DIR/nanochat/record_snapshots.py" nanochat/record_snapshots.py
source .venv/bin/activate
python "$PACKAGE_DIR/tools/apply_record_patch.py" "$REPO"
python -m py_compile scripts/base_train.py scripts/base_eval.py nanochat/record_snapshots.py
python -m nanochat.record_snapshots
python "$PACKAGE_DIR/record_tools/merge_record_snapshots.py" --selftest
cat > RECORD_ATTEMPT_PROVENANCE.json <<EOF
{
  "upstream": "https://github.com/karpathy/nanochat",
  "pull_request": 830,
  "commit": "$PR_COMMIT",
  "record_package": "nanochat_record_attempt_v1_1"
}
EOF
echo "record checkout repair PASS"
echo "repo=$REPO"
echo "commit=$ACTUAL"
git status --short
