#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
GROUP_NAME="${GROUP_NAME:-paper_main_d12_v1}"
ROOT="${ROOT_OVERRIDE:-$REPO/outputs/optimization_paper/$GROUP_NAME}"
OUT="${OUT:-paper_main_d12_v1_results.tar.gz}"
[[ -d "$ROOT" ]] || { echo "missing root: $ROOT" >&2; exit 1; }
[[ -f "$ROOT/figures/DIGEST.txt" ]] || { echo "missing merged figures/DIGEST.txt; suite may not be complete" >&2; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
DST="$TMP/$GROUP_NAME"
mkdir -p "$DST"
for f in SUBMISSION.txt FROZEN.env; do [[ -f "$ROOT/$f" ]] && cp "$ROOT/$f" "$DST/"; done
cp -a "$ROOT/figures" "$DST/"
mkdir -p "$DST/runs"
while IFS= read -r f; do
  rel="${f#$ROOT/}"
  mkdir -p "$DST/$(dirname "$rel")"
  cp "$f" "$DST/$rel"
done < <(find "$ROOT/runs" -type f \( -name 'result.json' -o -name 'recipe_evaluations.csv' -o -name 'training_trace.csv' \) 2>/dev/null | sort)
tar -C "$TMP" -czf "$OUT" "$GROUP_NAME"
sha256sum "$OUT" > "$OUT.sha256"
echo "wrote $OUT"
echo "wrote $OUT.sha256"
