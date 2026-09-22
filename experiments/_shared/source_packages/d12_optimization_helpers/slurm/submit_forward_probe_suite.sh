#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")"&&pwd)";REPO="${REPO:-$(cd "$D/.."&&pwd)}";SUB="$REPO/sub.sh"
RUN_NAME="${RUN_NAME:-forward_probe_screen}";ROOT="${ROOT:-$REPO/outputs/optimization_paper/$RUN_NAME}";ARMS_FILE="${ARMS_FILE:-$REPO/configs/forward_probe_arms_screen.txt}";MAX_PARALLEL="${MAX_PARALLEL:-8}"
N=$(grep -vcE '^\s*(#|$)' "$ARMS_FILE");mkdir -p "$ROOT" "$REPO/logs";PACK="${REUSE_PACK:-$ROOT/pack.pt}";export ROOT ARMS_FILE PACK
EXP="ALL,REPO=$REPO,ROOT=$ROOT,ARMS_FILE=$ARMS_FILE,PACK=$PACK"
if [[ -n "${REUSE_PACK:-}" ]];then PREP=reused;ARRAY=$($SUB --parsable --array="0-$((N-1))%$MAX_PARALLEL" --export="$EXP" "$D/95_forward_probe_array.sh");else PREP=$($SUB --parsable --export="$EXP" "$D/90_prepare_opt_pack.sh");ARRAY=$($SUB --parsable --dependency="afterok:$PREP" --array="0-$((N-1))%$MAX_PARALLEL" --export="$EXP" "$D/95_forward_probe_array.sh");fi
MERGE=$($SUB --parsable --dependency="afterany:$ARRAY" --export="$EXP" "$D/96_merge_forward_probe.sh")
echo "run=$RUN_NAME arms=$N prepare=$PREP array=$ARRAY merge=$MERGE outputs=$ROOT"
