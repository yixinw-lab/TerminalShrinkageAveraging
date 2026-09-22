#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
cd "$REPO"
if [[ -f "$REPO/slurm/_skip_env.sh" ]]; then
  # Reuse the same NanoChat environment that the Slurm jobs use.
  source "$REPO/slurm/_skip_env.sh"
fi
source "$REPO/slurm/_paper_python.sh"
echo "paper python: $PAPER_PYTHON ($($PAPER_PYTHON -V 2>&1))"
for f in \
  nanochat_meta/paper_main_d12.py \
  nanochat_meta/skip_core.py \
  nanochat_meta/skip_suite.py \
  nanochat_meta/filter_suite.py \
  nanochat_meta/trajectory_groups.py \
  slurm/_paper_python.sh \
  slurm/97_paper_main_d12_array.sh \
  slurm/99_merge_paper_main_d12.sh \
  slurm/submit_paper_main_d12.sh \
  slurm/97_paper_main_d12_confirm_array.sh \
  slurm/99_merge_paper_main_d12_confirmation.sh \
  slurm/submit_paper_main_d12_confirmation.sh \
  configs/paper_main_floors.txt \
  configs/paper_main_alpha_grid.txt \
  configs/paper_main_curve_steps.txt \
  configs/paper_main_confirmation_seeds.txt; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done
PACK="${PACK_OVERRIDE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
[[ -f "$PACK" ]] || { echo "missing D12 pack: $PACK" >&2; exit 1; }
"$PAPER_PYTHON" -m py_compile nanochat_meta/paper_main_d12.py
"$PAPER_PYTHON" -m nanochat_meta.paper_main_d12 selftest
"$PAPER_PYTHON" - <<'PY'
from pathlib import Path
floors=[x.strip() for x in Path('configs/paper_main_floors.txt').read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
assert floors == ['0.05','0.10','0.125','0.15','0.175'], floors
alpha=Path('configs/paper_main_alpha_grid.txt').read_text().strip()
vals=[float(x) for x in alpha.split(',')]
assert vals[0] == 0 and vals[-1] == 1 and 0.6 in vals
steps=[int(x) for x in Path('configs/paper_main_curve_steps.txt').read_text().strip().split(',')]
assert all(s % 16 == 8 for s in steps), [s for s in steps if s%16 != 8]
assert steps[-1] == 3000
print('config preflight PASS floors=5 alpha_grid=%d curve_points=%d' % (len(vals),len(steps)))
PY
echo "paper-main D12 suite verification PASS"
echo "pack=$PACK"
echo "selection split: curve32 / alpha_select64 / floor_select64 / holdout96"
echo "stage A: 5 one-GPU Muon+AdamW floor arms + 1 merge/plot job"
echo "stage B optional: 5 paired data-order repetitions x 2 frozen schedules"
