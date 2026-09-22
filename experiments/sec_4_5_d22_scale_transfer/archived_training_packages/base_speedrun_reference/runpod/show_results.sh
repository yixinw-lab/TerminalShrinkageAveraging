#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
find "$ROOT/results" -maxdepth 2 -name DIGEST.txt -print | sort | while read -r f; do echo; echo "===== $f ====="; cat "$f"; done
