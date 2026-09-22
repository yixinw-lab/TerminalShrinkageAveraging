#!/usr/bin/env bash
set -euo pipefail
WORK_ROOT="${WORK_ROOT:-/work/nvme/bhji/$USER/ExtrapProj/d22_delta_replication_v1}"
RESULT_ROOT="${RESULT_ROOT:-$WORK_ROOT/results}"
OUT="${1:-$HOME/ExtrapProj/d22_delta_replication_v1_compact_results.tar.gz}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/d22_delta_replication_v1"
for f in EXPERIMENT_FROZEN.txt MANIFEST.csv CODE_SHA256.txt PATCH_MANIFEST.txt; do [[ -f "$WORK_ROOT/$f" ]] && cp "$WORK_ROOT/$f" "$TMP/d22_delta_replication_v1/"; done
[[ -f "$RESULT_ROOT/SUBMISSION.txt" ]] && cp "$RESULT_ROOT/SUBMISSION.txt" "$TMP/d22_delta_replication_v1/"
[[ -d "$RESULT_ROOT/analysis" ]] && cp -a "$RESULT_ROOT/analysis" "$TMP/d22_delta_replication_v1/"
mkdir -p "$TMP/d22_delta_replication_v1/runs"
for d in "$RESULT_ROOT"/runs/*; do
  [[ -d "$d" ]] || continue
  b="$TMP/d22_delta_replication_v1/runs/$(basename "$d")"; mkdir -p "$b"
  for f in metrics.json RUN_FROZEN.txt SHA256.txt; do [[ -f "$d/$f" ]] && cp "$d/$f" "$b/"; done
  [[ -f "$d/torchrun.log" ]] && { grep -E 'GPU:|Calculated number of iterations|Physical stop step|POSTHOC|Total training time|Peak memory|Saved model-only' "$d/torchrun.log" > "$b/log_digest.txt" || true; }
done
tar -C "$TMP" -czf "$OUT" d22_delta_replication_v1
sha256sum "$OUT" > "$OUT.sha256"
echo "WROTE=$OUT"
echo "WROTE=$OUT.sha256"
