#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
cd "$REPO"
OUT="optimization_paper_small_results.tar.gz"
TMP=$(mktemp)
find outputs/optimization_paper -type f \
  \( -name 'result.json' -o -name '*.csv' -o -name '*.json' -o -name '*.png' -o -name '*.pdf' -o -name 'DIGEST.txt' \) \
  -print 2>/dev/null | sort > "$TMP"
if [[ ! -s "$TMP" ]]; then
  echo "no small result files below outputs/optimization_paper" >&2
  exit 1
fi
tar -czf "$OUT" -T "$TMP" \
  configs/averaging_recipes_mechanism.txt \
  configs/averaging_recipes_leaderboard.txt \
  configs/muon_filter_arms_mechanism.txt \
  configs/forward_probe_arms_screen.txt \
  2>/dev/null || tar -czf "$OUT" -T "$TMP"
rm -f "$TMP"
echo "wrote $REPO/$OUT"
ls -lh "$OUT"
