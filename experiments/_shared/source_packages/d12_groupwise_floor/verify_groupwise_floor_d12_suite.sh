#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
need=(
  nanochat_meta/groupwise_floor_d12.py
  tools/analyze_groupwise_floor_d12.py
  configs/groupwise_floor_manifest.csv
  configs/groupwise_floor_protocol.json
  slurm/_groupwise_floor_env.sh
  slurm/90_groupwise_floor_smoke.sh
  slurm/91_groupwise_floor_array.sh
  slurm/92_groupwise_floor_merge.sh
  slurm/submit_groupwise_floor_d12.sh
  pack_groupwise_floor_d12_results.sh
  README_GROUPWISE_FLOOR_D12.md
  VERSION_GROUPWISE_FLOOR_D12.txt
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

MANIFEST="$REPO/configs/groupwise_floor_manifest.csv"
grep -q $'\r' "$MANIFEST" && { echo "CRLF detected in $MANIFEST" >&2; exit 2; } || true
[[ "$(head -1 "$MANIFEST")" == "arm,stream_seed" ]] || { echo "bad manifest header" >&2; exit 2; }
N=$(( $(wc -l < "$MANIFEST") - 1 ))
[[ "$N" -eq 32 ]] || { echo "expected 32 manifest rows, got $N" >&2; exit 2; }
for arm in uniform05 uniform10 matched_uniform mapped; do
  C=$(awk -F, -v a="$arm" 'NR>1 && $1==a{n++} END{print n+0}' "$MANIFEST")
  [[ "$C" -eq 8 ]] || { echo "expected 8 rows for $arm, got $C" >&2; exit 2; }
done

python - "$REPO/configs/groupwise_floor_protocol.json" <<'PY'
import hashlib, json, sys
p=sys.argv[1]
obj=json.load(open(p))
claimed=obj.pop('protocol_sha256')
actual=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert claimed==actual, (claimed,actual)
assert obj['suite']=='d12_tsa_groupwise_floor_v1'
assert obj['arms']==['uniform05','uniform10','matched_uniform','mapped']
assert len(obj['stream_seeds'])==8
print('protocol_sha256=',claimed)
PY

for script in \
  "$REPO/slurm/_groupwise_floor_env.sh" \
  "$REPO/slurm/90_groupwise_floor_smoke.sh" \
  "$REPO/slurm/91_groupwise_floor_array.sh" \
  "$REPO/slurm/92_groupwise_floor_merge.sh" \
  "$REPO/slurm/submit_groupwise_floor_d12.sh" \
  "$REPO/pack_groupwise_floor_d12_results.sh" \
  "$REPO/verify_groupwise_floor_d12_suite.sh"; do
  bash -n "$script"
done

export REPO
source "$REPO/slurm/_groupwise_floor_env.sh"
echo "verification_python=$PAPER_PYTHON"
"$PAPER_PYTHON" --version
"$PAPER_PYTHON" - <<'PY'
import sys, torch, rustbpe, nanochat
print('torch=', torch.__version__)
print('rustbpe=', rustbpe.__file__)
print('nanochat=', nanochat.__file__)
print('python=', sys.executable)
PY
"$PAPER_PYTHON" -m py_compile \
  "$REPO/nanochat_meta/groupwise_floor_d12.py" \
  "$REPO/tools/analyze_groupwise_floor_d12.py"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PAPER_PYTHON" -m nanochat_meta.groupwise_floor_d12 selftest

PACK_NATIVE="${PACK_NATIVE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
[[ -f "$PACK_NATIVE" ]] || { echo "missing native pack: $PACK_NATIVE" >&2; exit 2; }
echo "native_pack=$PACK_NATIVE"

if [[ -f "$REPO/PACKAGE_MANIFEST_GROUPWISE_FLOOR_D12.sha256" ]]; then
  (cd "$REPO" && sha256sum -c PACKAGE_MANIFEST_GROUPWISE_FLOOR_D12.sha256)
fi

echo "GROUPWISE FLOOR D12 VERIFY PASS"
