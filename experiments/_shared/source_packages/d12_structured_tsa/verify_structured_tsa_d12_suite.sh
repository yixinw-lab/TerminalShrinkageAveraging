#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
for f in \
  "$REPO/nanochat_meta/paper_main_d12.py" \
  "$REPO/nanochat_meta/filter_suite.py" \
  "$REPO/nanochat_meta/skip_core.py" \
  "$REPO/nanochat_meta/skip_suite.py" \
  "$REPO/nanochat_meta/trajectory_groups.py" \
  "$REPO/nanochat_meta/structured_tsa_d12.py" \
  "$REPO/tools/freeze_structured_tsa_rules.py" \
  "$REPO/tools/analyze_structured_tsa_d12.py" \
  "$REPO/slurm/_skip_env.sh" \
  "$REPO/slurm/_paper_python.sh" \
  "$REPO/slurm/90_structured_tsa_smoke.sh" \
  "$REPO/slurm/91_structured_tsa_dev_array.sh" \
  "$REPO/slurm/92_structured_tsa_confirm_train_array.sh" \
  "$REPO/slurm/93_structured_tsa_freeze.sh" \
  "$REPO/slurm/94_structured_tsa_confirm_eval_array.sh" \
  "$REPO/slurm/95_structured_tsa_merge.sh" \
  "$REPO/slurm/submit_structured_tsa_d12.sh" \
  "$REPO/configs/structured_tsa_dev_manifest.csv" \
  "$REPO/configs/structured_tsa_confirm_manifest.csv" \
  "$REPO/sub.sh"; do
  [[ -e "$f" ]] || { echo "missing required file: $f" >&2; exit 2; }
done

source "$REPO/slurm/_skip_env.sh"
source "$REPO/slurm/_paper_python.sh"
"$PAPER_PYTHON" -m py_compile \
  "$REPO/nanochat_meta/structured_tsa_d12.py" \
  "$REPO/tools/freeze_structured_tsa_rules.py" \
  "$REPO/tools/analyze_structured_tsa_d12.py"
"$PAPER_PYTHON" -m nanochat_meta.structured_tsa_d12 selftest
"$PAPER_PYTHON" "$REPO/tools/freeze_structured_tsa_rules.py" --selftest
"$PAPER_PYTHON" "$REPO/tools/analyze_structured_tsa_d12.py" --selftest

bash -n \
  "$REPO/discover_structured_tsa_assets.sh" \
  "$REPO/pack_structured_tsa_d12_results.sh" \
  "$REPO/verify_structured_tsa_d12_suite.sh" \
  "$REPO/slurm/90_structured_tsa_smoke.sh" \
  "$REPO/slurm/91_structured_tsa_dev_array.sh" \
  "$REPO/slurm/92_structured_tsa_confirm_train_array.sh" \
  "$REPO/slurm/93_structured_tsa_freeze.sh" \
  "$REPO/slurm/94_structured_tsa_confirm_eval_array.sh" \
  "$REPO/slurm/95_structured_tsa_merge.sh" \
  "$REPO/slurm/submit_structured_tsa_d12.sh"

"$PAPER_PYTHON" - "$REPO" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path
repo = Path(sys.argv[1])

def load(name):
    with (repo / 'configs' / name).open(newline='') as h:
        return list(csv.DictReader(h))

dev = load('structured_tsa_dev_manifest.csv')
conf = load('structured_tsa_confirm_manifest.csv')
assert len(dev) == 10, len(dev)
assert len(conf) == 30, len(conf)
assert len({tuple(sorted(r.items())) for r in dev}) == len(dev)
assert len({tuple(sorted(r.items())) for r in conf}) == len(conf)
assert Counter(r['optimizer'] for r in dev) == {'native': 5, 'pure_adamw': 5}
assert set(r['floor'] for r in dev) == {'0.10'}
assert Counter((r['optimizer'], r['floor']) for r in conf) == {
    ('native', '0.05'): 5,
    ('native', '0.10'): 5,
    ('native', '0.15'): 5,
    ('pure_adamw', '0.05'): 5,
    ('pure_adamw', '0.10'): 5,
    ('pure_adamw', '0.15'): 5,
}
for opt in ('native', 'pure_adamw'):
    seed_sets = [
        {int(r['stream_seed']) for r in conf if r['optimizer'] == opt and r['floor'] == floor}
        for floor in ('0.05', '0.10', '0.15')
    ]
    assert seed_sets[0] == seed_sets[1] == seed_sets[2]
print('manifest verification PASS: 10 development + 30 confirmation trajectories')
PY

DEFAULT_NATIVE="$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt"
DEFAULT_ADAMW="$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/pure_adamw_seed1337/pack.pt"
if [[ -f "$DEFAULT_NATIVE" && -f "$DEFAULT_ADAMW" ]]; then
  echo "default native and pure-AdamW packs found"
else
  echo "WARNING: one or both default packs are missing. Run ./discover_structured_tsa_assets.sh '$REPO' and pass PACK_NATIVE/PACK_ADAMW overrides at submission." >&2
fi

echo "D12 structured TSA suite verification PASS"
echo "planned training trajectories: 40 (10 development + 30 confirmation)"
echo "planned post-hoc confirmation evaluations: 30"
