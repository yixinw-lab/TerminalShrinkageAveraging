#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
for f in \
  "$REPO/nanochat_meta/paper_main_d12.py" \
  "$REPO/nanochat_meta/appendix_averaging_d12.py" \
  "$REPO/tools/analyze_appendix_averaging.py" \
  "$REPO/slurm/97_appendix_averaging_array.sh" \
  "$REPO/slurm/99_merge_appendix_averaging.sh" \
  "$REPO/slurm/submit_appendix_averaging.sh" \
  "$REPO/configs/appendix_averaging_seeds.txt"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
source "$REPO/slurm/_skip_env.sh"
source "$REPO/slurm/_paper_python.sh"
"$PAPER_PYTHON" -m py_compile "$REPO/nanochat_meta/appendix_averaging_d12.py" "$REPO/tools/analyze_appendix_averaging.py"
"$PAPER_PYTHON" -m nanochat_meta.appendix_averaging_d12 selftest
bash -n "$REPO/slurm/97_appendix_averaging_array.sh" "$REPO/slurm/99_merge_appendix_averaging.sh" "$REPO/slurm/submit_appendix_averaging.sh"
echo 'D12 appendix averaging suite verification PASS'
