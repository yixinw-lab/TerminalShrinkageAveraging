#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?usage: package_results.sh /path/to/record_attempt_result_dir [output.tar.gz]}
OUT=${2:-record_attempt_small_results.tar.gz}
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
find -L "$ROOT" -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' -o -name '*.out' -o -name '*.err' \) \
  ! -name 'model_*.pt' ! -name 'optim_*.pt' ! -name 'snapshot_*.pt' \
  -print | sort > "$TMP"
[[ -s "$TMP" ]] || { echo "no small results under $ROOT" >&2; exit 1; }
tar -czf "$OUT" -T "$TMP"
echo "wrote $OUT"
ls -lh "$OUT"
