#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"
GROUP_NAME="${GROUP_NAME:-paper_main_d12_v1}"
ROOT="${GROUP_ROOT:-$REPO/outputs/optimization_paper/$GROUP_NAME}"
PACK="${PACK_OVERRIDE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
FLOORS_FILE="${FLOORS_FILE:-$REPO/configs/paper_main_floors.txt}"
ALPHA_GRID_FILE="${ALPHA_GRID_FILE:-$REPO/configs/paper_main_alpha_grid.txt}"
CURVE_STEPS_FILE="${CURVE_STEPS_FILE:-$REPO/configs/paper_main_curve_steps.txt}"

[[ -x "$SUB" ]] || { echo "missing executable submit wrapper: $SUB" >&2; exit 1; }
[[ -f "$PACK" ]] || { echo "missing paired D12 pack: $PACK" >&2; exit 1; }
[[ -f "$FLOORS_FILE" && -f "$ALPHA_GRID_FILE" && -f "$CURVE_STEPS_FILE" ]] || { echo "missing config file" >&2; exit 1; }
N=$(grep -vcE '^\s*(#|$)' "$FLOORS_FILE")
[[ "$N" -eq 5 ]] || { echo "expected exactly five floor arms, got $N" >&2; exit 1; }

if [[ -d "$ROOT" ]] && find -L "$ROOT" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  if [[ "${RESUME:-0}" != "1" ]]; then
    echo "refusing to reuse nonempty output directory: $ROOT" >&2
    echo "choose a new GROUP_NAME or inspect before setting RESUME=1" >&2
    exit 1
  fi
fi
mkdir -p "$ROOT" "$REPO/logs"
MAX_PARALLEL_VALUE="${MAX_PARALLEL:-4}"

MANIFEST="$ROOT/SUBMISSION.txt"
{
  echo "created_at=$(date -Is)"
  echo "design=main_paper_d12_sequential_alpha_then_floor_selection"
  echo "root=$ROOT"
  echo "pack=$PACK"
  echo "optimizer=native Muon+AdamW"
  echo "floors_file=$FLOORS_FILE"
  echo "alpha_grid_file=$ALPHA_GRID_FILE"
  echo "curve_steps_file=$CURVE_STEPS_FILE"
  echo "selection_protocol=curve32 / alpha_select64 / floor_select64 / holdout96"
  echo "alpha_selection=baseline 5% floor only"
  echo "floor_selection=frozen alpha on disjoint floor_select split"
  echo "headline_reporting=untouched holdout only"
  echo "primary_window=K8 spacing32"
  echo "endpoint_comparators=checkpoint EMA; SWA-style long late average; LAWA; adaptive tensor TSA"
  echo "max_parallel=$MAX_PARALLEL_VALUE"
} > "$MANIFEST"

export ROOT PACK FLOORS_FILE ALPHA_GRID_FILE CURVE_STEPS_FILE
EXP="ALL,REPO=$REPO,ROOT=$ROOT,PACK=$PACK,FLOORS_FILE=$FLOORS_FILE,ALPHA_GRID_FILE=$ALPHA_GRID_FILE,CURVE_STEPS_FILE=$CURVE_STEPS_FILE"
ARRAY=$(
  "$SUB" --parsable \
    --array="0-$((N-1))%$MAX_PARALLEL_VALUE" \
    --export="$EXP" \
    "$D/97_paper_main_d12_array.sh"
)
echo "array_job=$ARRAY" | tee -a "$MANIFEST"

MERGE=$(
  "$SUB" --parsable \
    --dependency="afterany:$ARRAY" \
    --export="ALL,REPO=$REPO,ROOT=$ROOT" \
    "$D/99_merge_paper_main_d12.sh"
)
echo "merge_job=$MERGE" | tee -a "$MANIFEST"

echo
echo "submitted D12 paper-main selection suite"
echo "group=$ROOT"
echo "array_job=$ARRAY"
echo "merge_job=$MERGE"
echo "monitor: squeue -u $USER -o '%.18i %.9P %.32j %.2t %.10M %.10l %R'"
echo "when done: cat $ROOT/figures/DIGEST.txt"
echo "frozen recipe: $ROOT/FROZEN.env"
