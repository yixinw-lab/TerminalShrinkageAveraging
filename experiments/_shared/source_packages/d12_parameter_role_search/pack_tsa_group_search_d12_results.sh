#!/bin/bash
set -euo pipefail
ROOT="${1:?usage: $0 /path/to/result_root [output.tar.gz]}"
OUT="${2:-$(basename "$ROOT")_summary.tar.gz}"
[[ -d "$ROOT" ]] || { echo "missing root: $ROOT" >&2; exit 2; }
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/summary"
for rel in \
  SUBMISSION.txt SMOKE.txt SMOKE_PARAMETER_MAP.csv \
  DEV_MANIFEST.csv CONFIRM_MANIFEST.csv SOURCE_MANIFEST.sha256 \
  FROZEN_GROUP_RULES.json DEV_CANDIDATE_SUMMARY.csv DEV_FAMILY_WINNERS.csv \
  figures/DIGEST.txt figures/confirmation_summary.csv \
  figures/confirmation_run_level.csv figures/key_contrasts.csv \
  figures/trajectory_role_stats.csv; do
  if [[ -f "$ROOT/$rel" ]]; then
    mkdir -p "$TMP/summary/$(dirname "$rel")"
    cp "$ROOT/$rel" "$TMP/summary/$rel"
  fi
done
tar -C "$TMP" -czf "$OUT" summary
echo "$OUT"
