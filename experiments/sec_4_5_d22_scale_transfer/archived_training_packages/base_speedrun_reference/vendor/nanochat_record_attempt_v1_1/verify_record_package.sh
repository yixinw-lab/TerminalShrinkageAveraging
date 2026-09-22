#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 -m py_compile \
  "$ROOT/nanochat/record_snapshots.py" \
  "$ROOT/tools/apply_record_patch.py" \
  "$ROOT/record_tools/merge_record_snapshots.py" \
  "$ROOT/record_tools/screen_candidates.py" \
  "$ROOT/record_tools/eval_candidate.py" \
  "$ROOT/record_tools/freeze_candidate.py" \
  "$ROOT/record_tools/summarize_confirmations.py"
for f in "$ROOT"/tools/*.sh "$ROOT"/slurm/*.sh "$ROOT"/slurm/*.sbatch; do bash -n "$f"; done
python3 "$ROOT/tools/apply_record_patch.py" --selftest
python3 "$ROOT/nanochat/record_snapshots.py"
python3 "$ROOT/record_tools/merge_record_snapshots.py" --selftest
python3 "$ROOT/tests/test_end_to_end.py"
grep -qx 'blend:0.587' "$ROOT/config/recipes_compact.txt"
grep -qx 'adaptive:hybrid' "$ROOT/config/recipes_compact.txt"
echo "nanochat record attempt package verification PASS"
