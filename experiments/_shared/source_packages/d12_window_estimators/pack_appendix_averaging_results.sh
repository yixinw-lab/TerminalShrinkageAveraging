#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
GROUP_NAME="${GROUP_NAME:-d12_appendix_averaging_v1}"
ROOT="$REPO/outputs/optimization_paper/$GROUP_NAME"
OUT="${OUT:-$REPO/${GROUP_NAME}_results.tar.gz}"
[[ -d "$ROOT" ]] || { echo "missing $ROOT" >&2; exit 2; }
tar -czf "$OUT" -C "$REPO/outputs/optimization_paper" "$GROUP_NAME"
sha256sum "$OUT" > "$OUT.sha256"
echo "WROTE=$OUT"
