#!/usr/bin/env bash
set -euo pipefail
DIR=$1 OUT=$2
mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$(dirname "$DIR")" "$(basename "$DIR")"
sha256sum "$OUT" > "$OUT.sha256"
echo "archive=$OUT"
cat "$OUT.sha256"
