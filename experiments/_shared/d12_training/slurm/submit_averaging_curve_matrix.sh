#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"
MODE="${MODE:-main}"
GROUP_NAME="${GROUP_NAME:-averaging_curves_d12_s5_v1}"
GROUP_ROOT="${GROUP_ROOT:-$REPO/outputs/optimization_paper/$GROUP_NAME}"
RECIPES_FILE="${RECIPES_FILE:-$REPO/configs/averaging_curve_recipes.txt}"
TRAJECTORIES_FILE="${TRAJECTORIES_FILE:-$REPO/configs/averaging_curve_trajectory.txt}"

case "$MODE" in
  smoke)
    DEPTH_VALUE="${DEPTH:-6}"
    SEEDS_VALUE="${SEEDS:-1337}"
    COMPUTE_VALUE="${COMPUTE_BUDGET:-500}"
    STREAM_VALUE="${STREAM_EQUIV:-700}"
    TOTAL_VALUE="${TOTAL_ITERATIONS:-500}"
    POINTS_VALUE="${RECIPE_EVAL_EQUIVS:-300,400,500}"
    RAW_EVERY_VALUE="${EVAL_EVERY_EQUIV:-100}"
    PACK_EVAL_VALUE="${EVAL_BATCHES:-32}"
    CURVE_EVAL_VALUE="${CURVE_EVAL_BATCHES:-8}"
    FINAL_EVAL_VALUE="${FINAL_EVAL_BATCHES:-32}"
    ;;
  main)
    DEPTH_VALUE="${DEPTH:-12}"
    SEEDS_VALUE="${SEEDS:-1337,2337,3337,4337,5337}"
    COMPUTE_VALUE="${COMPUTE_BUDGET:-3000}"
    STREAM_VALUE="${STREAM_EQUIV:-3200}"
    TOTAL_VALUE="${TOTAL_ITERATIONS:-3000}"
    POINTS_VALUE="${RECIPE_EVAL_EQUIVS:-600,900,1200,1500,1800,2100,2400,2700,3000}"
    RAW_EVERY_VALUE="${EVAL_EVERY_EQUIV:-300}"
    PACK_EVAL_VALUE="${EVAL_BATCHES:-256}"
    CURVE_EVAL_VALUE="${CURVE_EVAL_BATCHES:-32}"
    FINAL_EVAL_VALUE="${FINAL_EVAL_BATCHES:-256}"
    ;;
  *)
    echo "MODE must be smoke or main, got: $MODE" >&2
    exit 2
    ;;
esac

if [[ -d "$GROUP_ROOT" ]] && find -L "$GROUP_ROOT" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  if [[ "${RESUME:-0}" != "1" ]]; then
    echo "refusing to reuse nonempty output directory: $GROUP_ROOT" >&2
    echo "set RESUME=1 only after inspecting existing jobs/results" >&2
    exit 1
  fi
fi
mkdir -p "$GROUP_ROOT" "$REPO/logs"

IFS=',' read -r -a SEED_ARRAY <<<"$SEEDS_VALUE"
OPTIMIZERS=(pure_adamw native)
MERGE_IDS=()
MANIFEST="$GROUP_ROOT/SUBMISSION.txt"
{
  echo "created_at=$(date -Is)"
  echo "mode=$MODE"
  echo "group_root=$GROUP_ROOT"
  echo "depth=$DEPTH_VALUE"
  echo "seeds=$SEEDS_VALUE"
  echo "optimizers=${OPTIMIZERS[*]}"
  echo "compute_budget=$COMPUTE_VALUE"
  echo "recipe_eval_equivs=$POINTS_VALUE"
  echo "recipes_file=$RECIPES_FILE"
  echo "trajectory_file=$TRAJECTORIES_FILE"
} > "$MANIFEST"

for optimizer in "${OPTIMIZERS[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -n "$seed" ]] || continue
    RUN_NAME="$GROUP_NAME/${optimizer}_seed${seed}"
    ROOT="$REPO/outputs/optimization_paper/$RUN_NAME"

    echo "submitting optimizer=$optimizer seed=$seed root=$ROOT"
    output=$(
      RUN_NAME="$RUN_NAME" \
      ROOT="$ROOT" \
      TRAJECTORIES_FILE="$TRAJECTORIES_FILE" \
      RECIPES_FILE="$RECIPES_FILE" \
      OPTIMIZER="$optimizer" \
      DEPTH="$DEPTH_VALUE" \
      PREFIX_STEPS=0 \
      PREFIX_LR_SCALE=1.0 \
      PACK_WARMDOWN_RATIO=0 \
      COMPUTE_BUDGET="$COMPUTE_VALUE" \
      STREAM_EQUIV="$STREAM_VALUE" \
      TOTAL_ITERATIONS="$TOTAL_VALUE" \
      RECIPE_EVAL_EQUIVS="$POINTS_VALUE" \
      EVAL_EVERY_EQUIV="$RAW_EVERY_VALUE" \
      EVAL_BATCHES="$PACK_EVAL_VALUE" \
      CURVE_EVAL_BATCHES="$CURVE_EVAL_VALUE" \
      FINAL_EVAL_BATCHES="$FINAL_EVAL_VALUE" \
      SNAPSHOT_EVERY=4 \
      SNAPSHOT_DTYPE=bfloat16 \
      MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}" \
      DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}" \
      GRAD_ACCUM="${GRAD_ACCUM:-16}" \
      MAX_PARALLEL=1 \
      SEED="$seed" \
      bash "$D/submit_averaging_curve_single.sh"
    )
    echo "$output"
    echo "$output" >> "$MANIFEST"
    merge_id=$(sed -n 's/.* merge=\([^ ]*\).*/\1/p' <<<"$output" | tail -1)
    if [[ -z "$merge_id" ]]; then
      echo "could not parse merge job id from: $output" >&2
      exit 1
    fi
    MERGE_IDS+=("$merge_id")
  done
done

if [[ "${#MERGE_IDS[@]}" -eq 0 ]]; then
  echo "no jobs submitted" >&2
  exit 1
fi

dependency=$(IFS=:; echo "${MERGE_IDS[*]}")
export GROUP_ROOT
FINAL_JOB=$(
  "$SUB" --parsable \
    --dependency="afterany:$dependency" \
    --export="ALL,REPO=$REPO,GROUP_ROOT=$GROUP_ROOT" \
    "$D/99_merge_averaging_curves.sh"
)
echo "final_plot_job=$FINAL_JOB" | tee -a "$MANIFEST"

echo
echo "submitted ${#MERGE_IDS[@]} optimizer/seed runs"
echo "group=$GROUP_ROOT"
echo "final_plot_job=$FINAL_JOB"
echo "monitor: squeue -u $USER -o '%.18i %.9P %.32j %.2t %.10M %.10l %R'"
echo "manifest: $MANIFEST"
