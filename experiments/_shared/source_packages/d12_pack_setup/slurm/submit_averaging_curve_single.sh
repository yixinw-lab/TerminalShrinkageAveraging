#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"

RUN_NAME="${RUN_NAME:-averaging_curve_single}"
ROOT="${ROOT:-$REPO/outputs/optimization_paper/$RUN_NAME}"
TRAJECTORIES_FILE="${TRAJECTORIES_FILE:-$REPO/configs/averaging_curve_trajectory.txt}"
RECIPES_FILE="${RECIPES_FILE:-$REPO/configs/averaging_curve_recipes.txt}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
N=$(grep -vcE '^\s*(#|$)' "$TRAJECTORIES_FILE")

if [[ ! -x "$SUB" ]]; then
  echo "missing executable submit wrapper: $SUB" >&2
  exit 1
fi
if [[ "$N" -lt 1 ]]; then
  echo "no trajectory rows in $TRAJECTORIES_FILE" >&2
  exit 1
fi

mkdir -p "$ROOT" "$REPO/logs"
export ROOT TRAJECTORIES_FILE RECIPES_FILE
EXP="ALL,REPO=$REPO,ROOT=$ROOT,TRAJECTORIES_FILE=$TRAJECTORIES_FILE,RECIPES_FILE=$RECIPES_FILE"

if [[ -n "${REUSE_PACK:-}" ]]; then
  PACK="$REUSE_PACK"
  PREP="reused"
else
  PACK="$ROOT/pack.pt"
  export PACK
  EXP="$EXP,PACK=$PACK"
  PREP=$($SUB --parsable --export="$EXP" "$D/90_prepare_opt_pack.sh")
fi
export PACK
if [[ "$EXP" != *",PACK="* ]]; then
  EXP="$EXP,PACK=$PACK"
fi

if [[ "$PREP" == "reused" ]]; then
  ARRAY=$($SUB --parsable \
    --array="0-$((N-1))%$MAX_PARALLEL" \
    --export="$EXP" \
    "$D/97_averaging_curve_array.sh")
else
  ARRAY=$($SUB --parsable \
    --dependency="afterok:$PREP" \
    --array="0-$((N-1))%$MAX_PARALLEL" \
    --export="$EXP" \
    "$D/97_averaging_curve_array.sh")
fi

MERGE=$($SUB --parsable \
  --dependency="afterany:$ARRAY" \
  --export="$EXP" \
  "$D/92_merge_muon_avg.sh")

echo "run=$RUN_NAME trajectories=$N prepare=$PREP array=$ARRAY merge=$MERGE outputs=$ROOT"
