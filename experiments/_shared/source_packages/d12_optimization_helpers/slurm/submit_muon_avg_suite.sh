#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; REPO="${REPO:-$(cd "$D/.." && pwd)}"; SUB="$REPO/sub.sh"
RUN_NAME="${RUN_NAME:-muon_avg_mechanism}"; ROOT="${ROOT:-$REPO/outputs/optimization_paper/$RUN_NAME}"
TRAJECTORIES_FILE="${TRAJECTORIES_FILE:-$REPO/configs/trajectory_configs_mechanism.txt}"
RECIPES_FILE="${RECIPES_FILE:-$REPO/configs/averaging_recipes_mechanism.txt}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"; N=$(grep -vcE '^\s*(#|$)' "$TRAJECTORIES_FILE")
mkdir -p "$ROOT" "$REPO/logs"; export ROOT TRAJECTORIES_FILE RECIPES_FILE
EXP="ALL,REPO=$REPO,ROOT=$ROOT,TRAJECTORIES_FILE=$TRAJECTORIES_FILE,RECIPES_FILE=$RECIPES_FILE"
if [[ -n "${REUSE_PACK:-}" ]]; then PACK="$REUSE_PACK";export PACK;EXP="$EXP,PACK=$PACK";PREP=reused;else PACK="$ROOT/pack.pt";export PACK;EXP="$EXP,PACK=$PACK";PREP=$($SUB --parsable --export="$EXP" "$D/90_prepare_opt_pack.sh");fi
if [[ "$PREP" == reused ]]; then
  ARRAY=$($SUB --parsable --array="0-$((N-1))%$MAX_PARALLEL" --export="$EXP" "$D/91_muon_avg_array.sh")
else
  ARRAY=$($SUB --parsable --dependency="afterok:$PREP" --array="0-$((N-1))%$MAX_PARALLEL" --export="$EXP" "$D/91_muon_avg_array.sh")
fi
MERGE=$($SUB --parsable --dependency="afterany:$ARRAY" --export="$EXP" "$D/92_merge_muon_avg.sh")
echo "run=$RUN_NAME trajectories=$N prepare=$PREP array=$ARRAY merge=$MERGE outputs=$ROOT"
