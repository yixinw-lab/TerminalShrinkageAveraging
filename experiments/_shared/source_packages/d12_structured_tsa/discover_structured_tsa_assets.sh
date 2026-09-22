#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
cd "$REPO"
echo "===== REPOSITORY ====="
pwd
if git rev-parse HEAD >/dev/null 2>&1; then
  git rev-parse HEAD
  git status --short
fi

echo
echo "===== EXPECTED PACKS ====="
for p in \
  outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt \
  outputs/optimization_paper/averaging_curves_d12_s5_v1/pure_adamw_seed1337/pack.pt; do
  if [[ -f "$p" ]]; then
    ls -lh "$p"
    readlink -f "$p" || true
  else
    echo "MISSING $p"
  fi
done

echo
echo "===== ALL CANDIDATE PACKS ====="
find outputs/optimization_paper -type f -name pack.pt -print 2>/dev/null | sort | sed -n '1,200p'

echo
echo "===== REQUIRED HELPER MODULES ====="
for f in \
  nanochat_meta/paper_main_d12.py \
  nanochat_meta/filter_suite.py \
  nanochat_meta/skip_core.py \
  nanochat_meta/skip_suite.py \
  nanochat_meta/trajectory_groups.py \
  slurm/_skip_env.sh \
  slurm/_paper_python.sh \
  sub.sh; do
  [[ -e "$f" ]] && echo "FOUND $f" || echo "MISSING $f"
done

echo
echo "===== STORAGE ====="
quota 2>/dev/null || true
df -h "$REPO" "$REPO/outputs" 2>/dev/null || true
readlink -f "$REPO/outputs" 2>/dev/null || true


echo
echo "===== RECOMMENDED RESULT ROOT ====="
echo "/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper/d12_structured_tsa_v1"
echo "Use a different allocation code/path only if accounts or quota output shows that bhji is not your active work allocation."
