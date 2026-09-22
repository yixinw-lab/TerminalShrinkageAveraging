#!/bin/bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"
OUT="${OUT:-optimization_paper_small_results.tar.gz}"
ROOT="${RESULT_ROOT:-outputs/optimization_paper}"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

if [[ ! -e "$ROOT" ]]; then
  echo "missing $REPO/$ROOT" >&2
  exit 1
fi

# outputs/optimization_paper is intentionally a symlink to NVMe on DeltaAI.
# GNU find does not follow a command-line symlink by default, so use -L.
find -L "$ROOT" -type f \
  \( -name 'result.json' \
     -o -name '*.csv' \
     -o -name '*.json' \
     -o -name '*.png' \
     -o -name '*.pdf' \
     -o -name '*.txt' \
     -o -name '*.md' \) \
  ! -name 'pack.pt' \
  ! -name '*.pt' \
  ! -name '*.pth' \
  ! -name '*.bin' \
  ! -name '*.safetensors' \
  -print 2>/dev/null | sort > "$TMP"

if [[ ! -s "$TMP" ]]; then
  echo "no small result files below $(readlink -f "$ROOT" 2>/dev/null || echo "$ROOT")" >&2
  exit 1
fi

tar -czf "$OUT" -T "$TMP"
echo "wrote $REPO/$OUT"
echo "files=$(wc -l < "$TMP")"
ls -lh "$OUT"
