#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"
GROUP_NAME="${GROUP_NAME:-d12_appendix_averaging_v1}"
ROOT="${GROUP_ROOT:-$REPO/outputs/optimization_paper/$GROUP_NAME}"
PACK="${PACK_OVERRIDE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
SEEDS_FILE="${SEEDS_FILE:-$REPO/configs/appendix_averaging_seeds.txt}"
[[ -x "$SUB" && -f "$PACK" && -f "$SEEDS_FILE" ]] || { echo 'missing sub.sh, pack, or seeds file' >&2; exit 2; }
mkdir -p "$ROOT" "$REPO/logs"
N=$(grep -vcE '^\s*(#|$)' "$SEEDS_FILE")
MAXP="${MAX_PARALLEL:-4}"
{
  echo "created_at=$(date -Is)"
  echo "design=five_stream_seed_D12_K_spacing_and_averaging_family_ablation"
  echo "physical_and_schedule_horizon=3000"
  echo "terminal_floor=0.10"
  echo "primary_alpha=0.55"
  echo "K_grid=4,8,16"
  echo "spacing_grid=16,32,64"
  echo "matched_EWA_betas=0.50,0.75,0.90,0.95"
  echo "historical_EMA=beta0.95_K16_s16"
  echo "SWA_style=K32_s16"
  echo "adaptive=hybrid_all_and_muon"
  echo "pack=$PACK"
} > "$ROOT/SUBMISSION.txt"
export ROOT PACK SEEDS_FILE
EXP="ALL,REPO=$REPO,ROOT=$ROOT,PACK=$PACK,SEEDS_FILE=$SEEDS_FILE"
ARRAY=$("$SUB" --parsable --array="0-$((N-1))%$MAXP" --export="$EXP" "$D/97_appendix_averaging_array.sh")
echo "array_job=$ARRAY" | tee -a "$ROOT/SUBMISSION.txt"
MERGE=$("$SUB" --parsable --dependency="afterok:$ARRAY" --export="ALL,REPO=$REPO,ROOT=$ROOT" "$D/99_merge_appendix_averaging.sh")
echo "merge_job=$MERGE" | tee -a "$ROOT/SUBMISSION.txt"
echo "submitted appendix averaging suite: array=$ARRAY merge=$MERGE root=$ROOT"
