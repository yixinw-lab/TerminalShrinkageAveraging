#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
GROUP_NAME="${GROUP_NAME:-d12_structured_tsa_v1}"
ROOT="${GROUP_ROOT:-${ROOT_OVERRIDE:-$REPO/outputs/optimization_paper/$GROUP_NAME}}"
OUT="${OUT:-$REPO/${GROUP_NAME}_compact_results.tar.gz}"
[[ -d "$ROOT" ]] || { echo "missing result root $ROOT" >&2; exit 2; }
[[ -f "$ROOT/figures/DIGEST.txt" ]] || { echo "merge output is missing: $ROOT/figures/DIGEST.txt" >&2; exit 2; }
PARENT="$(dirname "$ROOT")"
BASE="$(basename "$ROOT")"

if [[ "${INCLUDE_ARTIFACTS:-0}" == 1 ]]; then
  tar -czf "$OUT" -C "$PARENT" "$BASE"
else
  tar -czf "$OUT" \
    --exclude='*/terminal_artifact.pt' \
    -C "$PARENT" "$BASE"
fi
sha256sum "$OUT" > "$OUT.sha256"
echo "WROTE=$OUT"
echo "SOURCE_ROOT=$ROOT"
echo "SHA256=$(cut -d' ' -f1 "$OUT.sha256")"
