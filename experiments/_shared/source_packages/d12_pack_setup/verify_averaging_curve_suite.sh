#!/bin/bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

required=(
  nanochat_meta/muon_aware_averaging.py
  nanochat_meta/averaging_curve_plot.py
  configs/averaging_curve_recipes.txt
  configs/averaging_curve_trajectory.txt
  slurm/97_averaging_curve_array.sh
  slurm/99_merge_averaging_curves.sh
  slurm/submit_averaging_curve_single.sh
  slurm/submit_averaging_curve_matrix.sh
  slurm/90_prepare_opt_pack.sh
  slurm/92_merge_muon_avg.sh
  slurm/_skip_env.sh
  sub.sh
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing $path" >&2; exit 1; }
done

bash -n slurm/97_averaging_curve_array.sh
bash -n slurm/99_merge_averaging_curves.sh
bash -n slurm/submit_averaging_curve_single.sh
bash -n slurm/submit_averaging_curve_matrix.sh
bash -n pack_optimization_paper_results.sh

# Use the same Python environment as the Slurm jobs. The login shell itself may
# not expose a `python` command on DeltaAI.
source slurm/_skip_env.sh
python -m py_compile \
  nanochat_meta/muon_aware_averaging.py \
  nanochat_meta/averaging_curve_plot.py
python -m nanochat_meta.muon_aware_averaging selftest

echo "averaging curve suite verification PASS"
