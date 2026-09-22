#!/bin/bash
set -euo pipefail
ROOT="${1:?usage: $0 RESULT_ROOT [OUTPUT.tar.gz]}"
OUT="${2:-d12_tsa_groupwise_floor_v1_summary.tar.gz}"
[[ -f "$ROOT/figures/DIGEST.txt" ]] || { echo "missing merged result: $ROOT/figures/DIGEST.txt" >&2; exit 2; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/d12_tsa_groupwise_floor_v1"
for rel in \
  SUBMISSION.txt SOURCE_MANIFEST.sha256 PROTOCOL.json RUN_MANIFEST.csv SMOKE.txt \
  SMOKE_OPTIMIZER_GROUPS.csv SMOKE_SCHEDULE.csv \
  figures/DIGEST.txt figures/paired_contrasts.csv figures/arm_summary.csv \
  figures/per_seed.csv figures/groupwise_floor_summary.png figures/APPENDIX_DRAFT.md; do
  [[ -f "$ROOT/$rel" ]] && { mkdir -p "$TMP/d12_tsa_groupwise_floor_v1/$(dirname "$rel")"; cp "$ROOT/$rel" "$TMP/d12_tsa_groupwise_floor_v1/$rel"; }
done
find "$ROOT/runs" -name result.json -type f -print0 | while IFS= read -r -d '' f; do
  rel="${f#$ROOT/}"
  mkdir -p "$TMP/d12_tsa_groupwise_floor_v1/$(dirname "$rel")"
  cp "$f" "$TMP/d12_tsa_groupwise_floor_v1/$rel"
done
tar -C "$TMP" -czf "$OUT" d12_tsa_groupwise_floor_v1
echo "wrote $OUT"
