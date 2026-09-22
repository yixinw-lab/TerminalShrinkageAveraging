#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in tools/patch_base_train_d22.py tools/analyze_d22_delta.py configs/d22_delta_manifest.csv prepare_d22_delta_checkout.sh slurm/00_setup.sh slurm/01_smoke.sh slurm/02_train_array.sh slurm/03_merge.sh slurm/submit_d22_delta.sh; do
  [[ -f "$ROOT/$f" ]] || { echo "missing $f"; exit 1; }
done
python3 -m py_compile "$ROOT/tools/patch_base_train_d22.py" "$ROOT/tools/analyze_d22_delta.py"
for f in "$ROOT"/*.sh "$ROOT"/slurm/*.sh; do bash -n "$f"; done
[[ $(tail -n +2 "$ROOT/configs/d22_delta_manifest.csv" | wc -l) -eq 6 ]]
echo 'SUITE STATIC VERIFICATION PASS'
