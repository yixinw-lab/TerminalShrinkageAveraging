#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
need=(
  nanochat_meta/tsa_group_search_d12.py
  tools/analyze_tsa_group_search_d12.py
  configs/tsa_group_search_dev_manifest.csv
  configs/tsa_group_search_confirm_manifest.csv
  slurm/_tsa_group_search_env.sh
  slurm/90_tsa_group_search_smoke.sh
  slurm/91_tsa_group_search_dev_array.sh
  slurm/92_tsa_group_search_freeze.sh
  slurm/93_tsa_group_search_confirm_array.sh
  slurm/94_tsa_group_search_merge.sh
  slurm/submit_tsa_group_search_d12.sh
  pack_tsa_group_search_d12_results.sh
  README_TSA_GROUP_SEARCH_D12.md
  VERSION_TSA_GROUP_SEARCH_D12.txt
  nanochat_meta/structured_tsa_d12.py
  nanochat_meta/paper_main_d12.py
  nanochat_meta/filter_suite.py
  nanochat_meta/skip_suite.py
  slurm/_skip_env.sh
  sub.sh
)
for rel in "${need[@]}"; do
  [[ -f "$REPO/$rel" ]] || { echo "missing required file: $REPO/$rel" >&2; exit 2; }
done
[[ -x "$REPO/sub.sh" ]] || { echo "$REPO/sub.sh is not executable" >&2; exit 2; }

for manifest in "$REPO/configs/tsa_group_search_dev_manifest.csv" "$REPO/configs/tsa_group_search_confirm_manifest.csv"; do
  grep -q $'\r' "$manifest" && { echo "CRLF detected in $manifest" >&2; exit 2; } || true
  [[ "$(head -1 "$manifest")" == "optimizer,floor,stream_seed" ]] || { echo "bad manifest header: $manifest" >&2; exit 2; }
done
DEV_N=$(( $(wc -l < "$REPO/configs/tsa_group_search_dev_manifest.csv") - 1 ))
CONFIRM_N=$(( $(wc -l < "$REPO/configs/tsa_group_search_confirm_manifest.csv") - 1 ))
[[ "$DEV_N" -eq 8 ]] || { echo "expected 8 dev rows, got $DEV_N" >&2; exit 2; }
[[ "$CONFIRM_N" -eq 16 ]] || { echo "expected 16 confirm rows, got $CONFIRM_N" >&2; exit 2; }

for script in \
  "$REPO/slurm/_tsa_group_search_env.sh" \
  "$REPO/slurm/90_tsa_group_search_smoke.sh" \
  "$REPO/slurm/91_tsa_group_search_dev_array.sh" \
  "$REPO/slurm/92_tsa_group_search_freeze.sh" \
  "$REPO/slurm/93_tsa_group_search_confirm_array.sh" \
  "$REPO/slurm/94_tsa_group_search_merge.sh" \
  "$REPO/slurm/submit_tsa_group_search_d12.sh" \
  "$REPO/pack_tsa_group_search_d12_results.sh" \
  "$REPO/verify_tsa_group_search_d12_suite.sh"; do
  bash -n "$script"
done

export REPO
source "$REPO/slurm/_tsa_group_search_env.sh"
echo "verification_python=$PAPER_PYTHON"
"$PAPER_PYTHON" --version
"$PAPER_PYTHON" - <<'PY'
import sys, torch, rustbpe, nanochat
print("torch=", torch.__version__)
print("rustbpe=", rustbpe.__file__)
print("nanochat=", nanochat.__file__)
print("python=", sys.executable)
PY
"$PAPER_PYTHON" -m py_compile \
  "$REPO/nanochat_meta/tsa_group_search_d12.py" \
  "$REPO/tools/analyze_tsa_group_search_d12.py"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PAPER_PYTHON" -m nanochat_meta.tsa_group_search_d12 selftest
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PAPER_PYTHON" "$REPO/tools/analyze_tsa_group_search_d12.py" --selftest

PACK_NATIVE="${PACK_NATIVE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
PACK_ADAMW="${PACK_ADAMW:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/pure_adamw_seed1337/pack.pt}"
[[ -f "$PACK_NATIVE" ]] || { echo "missing native pack: $PACK_NATIVE" >&2; exit 2; }
[[ -f "$PACK_ADAMW" ]] || { echo "missing pure-AdamW pack: $PACK_ADAMW" >&2; exit 2; }

echo "native_pack=$PACK_NATIVE"
echo "pure_adamw_pack=$PACK_ADAMW"

if [[ -f "$REPO/PACKAGE_MANIFEST_TSA_GROUP_SEARCH_D12.sha256" ]]; then
  (cd "$REPO" && sha256sum -c PACKAGE_MANIFEST_TSA_GROUP_SEARCH_D12.sha256)
fi

echo "TSA GROUP SEARCH D12 VERIFY PASS"
