#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"
SOURCE_GROUP="${SOURCE_GROUP:-paper_main_d12_v1}"
FROZEN="${FROZEN:-$REPO/outputs/optimization_paper/$SOURCE_GROUP/FROZEN.env}"
GROUP_NAME="${GROUP_NAME:-paper_main_d12_confirm_v1}"
ROOT="${GROUP_ROOT:-$REPO/outputs/optimization_paper/$GROUP_NAME}"
PACK="${PACK_OVERRIDE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
SEEDS_FILE="${SEEDS_FILE:-$REPO/configs/paper_main_confirmation_seeds.txt}"
CURVE_STEPS_FILE="${CURVE_STEPS_FILE:-$REPO/configs/paper_main_curve_steps.txt}"
[[ -x "$SUB" && -f "$FROZEN" && -f "$PACK" && -f "$SEEDS_FILE" ]] || { echo "missing submit wrapper / frozen recipe / pack / seeds file" >&2; exit 1; }
# shellcheck disable=SC1090
source "$FROZEN"
: "${SELECTED_ALPHA:?}" "${SELECTED_FLOOR:?}"
mkdir -p "$ROOT" "$REPO/logs"
COMBOS_FILE="$ROOT/confirmation_combos.csv"
: > "$COMBOS_FILE"
while read -r seed; do
  [[ -z "$seed" || "$seed" =~ ^# ]] && continue
  echo "$seed,0.05,$SELECTED_ALPHA,baseline" >> "$COMBOS_FILE"
  echo "$seed,$SELECTED_FLOOR,$SELECTED_ALPHA,selected" >> "$COMBOS_FILE"
done < "$SEEDS_FILE"
N=$(wc -l < "$COMBOS_FILE")
[[ "$N" -gt 0 ]] || { echo "no confirmation combos" >&2; exit 1; }
MAX_PARALLEL_VALUE="${MAX_PARALLEL:-4}"
MANIFEST="$ROOT/SUBMISSION.txt"
{
  echo "created_at=$(date -Is)"
  echo "design=paired_data_order_confirmation_of_frozen_d12_recipe"
  echo "source_group=$SOURCE_GROUP"
  echo "frozen=$FROZEN"
  echo "selected_alpha=$SELECTED_ALPHA"
  echo "selected_floor=$SELECTED_FLOOR"
  echo "combos=$COMBOS_FILE"
  echo "note=same initialization and finite batch multiset; independent data-order permutations across repetitions"
} > "$MANIFEST"
export ROOT PACK FROZEN COMBOS_FILE CURVE_STEPS_FILE
EXP="ALL,REPO=$REPO,ROOT=$ROOT,PACK=$PACK,FROZEN=$FROZEN,COMBOS_FILE=$COMBOS_FILE,CURVE_STEPS_FILE=$CURVE_STEPS_FILE"
ARRAY=$("$SUB" --parsable --array="0-$((N-1))%$MAX_PARALLEL_VALUE" --export="$EXP" "$D/97_paper_main_d12_confirm_array.sh")
echo "array_job=$ARRAY" | tee -a "$MANIFEST"
MERGE=$("$SUB" --parsable --dependency="afterany:$ARRAY" --export="ALL,REPO=$REPO,ROOT=$ROOT,FROZEN=$FROZEN" "$D/99_merge_paper_main_d12_confirmation.sh")
echo "merge_job=$MERGE" | tee -a "$MANIFEST"
echo "submitted D12 frozen confirmation: array=$ARRAY merge=$MERGE root=$ROOT"
