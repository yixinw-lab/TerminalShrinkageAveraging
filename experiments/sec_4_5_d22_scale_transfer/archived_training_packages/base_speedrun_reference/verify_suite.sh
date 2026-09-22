#!/usr/bin/env bash
set -euo pipefail
SUITE=$(cd "$(dirname "$0")" && pwd)
find "$SUITE" -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python3 -m py_compile "$SUITE/tools/validate_treatment_manifest.py" "$SUITE/tools/summarize_arm.py"
python3 "$SUITE/vendor/nanochat_record_attempt_v1_1/record_tools/merge_record_snapshots.py" --selftest
bash "$SUITE/vendor/nanochat_record_attempt_v1_1/verify_record_package.sh"
echo "causal speedrun suite verification PASS"
