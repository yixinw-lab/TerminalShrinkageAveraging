#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
cd "$REPO"

for f in \
  nanochat_meta/skip_core.py \
  nanochat_meta/skip_suite.py \
  nanochat_meta/filter_suite.py \
  nanochat_meta/causal_fastforward.py \
  nanochat_meta/trajectory_groups.py \
  nanochat_meta/muon_aware_averaging.py \
  nanochat_meta/muon_selective_filter.py \
  nanochat_meta/forward_probe_fastforward.py; do
  [[ -f "$f" ]] || { echo "missing required file: $f" >&2; exit 2; }
done

if [[ -f slurm/_skip_env.sh ]]; then
  source slurm/_skip_env.sh
fi

python -m py_compile \
  nanochat_meta/trajectory_groups.py \
  nanochat_meta/muon_aware_averaging.py \
  nanochat_meta/muon_aware_plot.py \
  nanochat_meta/muon_selective_filter.py \
  nanochat_meta/muon_selective_filter_plot.py \
  nanochat_meta/forward_probe_fastforward.py \
  nanochat_meta/forward_probe_plot.py \
  speedrun/install_speedrun_lawa_hook.py \
  speedrun/create_speedrun_lawa_script.py \
  speedrun/average_lawa_snapshots.py

for f in \
  slurm/90_prepare_opt_pack.sh \
  slurm/91_muon_avg_array.sh \
  slurm/92_merge_muon_avg.sh \
  slurm/93_muon_filter_array.sh \
  slurm/94_merge_muon_filter.sh \
  slurm/95_forward_probe_array.sh \
  slurm/96_merge_forward_probe.sh \
  slurm/submit_muon_avg_suite.sh \
  slurm/submit_muon_filter_suite.sh \
  slurm/submit_forward_probe_suite.sh; do
  bash -n "$f"
done

python -m nanochat_meta.muon_aware_averaging selftest
python -m nanochat_meta.muon_selective_filter selftest
python -m nanochat_meta.forward_probe_fastforward selftest

python - <<'PY'
from pathlib import Path
from nanochat_meta.muon_aware_averaging import load_recipes, recipe_window_spacing
from nanochat_meta.muon_selective_filter import arm_requirements
from nanochat_meta.forward_probe_fastforward import parse_probe

for p in [
    Path('configs/averaging_recipes_smoke.txt'),
    Path('configs/averaging_recipes_mechanism.txt'),
    Path('configs/averaging_recipes_leaderboard.txt'),
]:
    for recipe in load_recipes(p):
        recipe_window_spacing(recipe)

for p in [Path('configs/muon_filter_arms_smoke.txt'), Path('configs/muon_filter_arms_mechanism.txt')]:
    for line in p.read_text().splitlines():
        line=line.strip()
        if line and not line.startswith('#'):
            arm_requirements(line)

for p in [Path('configs/forward_probe_arms_smoke.txt'), Path('configs/forward_probe_arms_screen.txt')]:
    for line in p.read_text().splitlines():
        line=line.strip()
        if line and not line.startswith('#') and not line.startswith(('exact_lr:','eval_lawa_lr:')):
            parse_probe(line)
print('all configured recipes/arms parse')
PY

echo "optimization paper suite verification PASS"
