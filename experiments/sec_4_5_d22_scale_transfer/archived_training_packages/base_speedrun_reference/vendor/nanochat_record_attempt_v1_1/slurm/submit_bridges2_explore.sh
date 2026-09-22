#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PACKAGE_DIR"
EXPLORE_ENV=${1:-"$PACKAGE_DIR/config/explore.env"}
SITE_ENV=${2:-"$PACKAGE_DIR/config/site_bridges2.env"}
[[ -f "$EXPLORE_ENV" ]] || { echo "missing $EXPLORE_ENV (copy from .example)" >&2; exit 1; }
[[ -f "$SITE_ENV" ]] || { echo "missing $SITE_ENV (copy from .example)" >&2; exit 1; }
source "$EXPLORE_ENV"; source "$SITE_ENV"
: "${BRIDGES2_ACCOUNT:?set BRIDGES2_ACCOUNT in site env}"
mkdir -p "$PACKAGE_DIR/logs"
JOB=$(sbatch --parsable \
  --account="$BRIDGES2_ACCOUNT" \
  --partition="${BRIDGES2_PARTITION:-GPU}" \
  --time="${BRIDGES2_TIME_EXPLORE:-08:00:00}" \
  --export=ALL,REPO="$REPO",NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",RECORD_PACKAGE_DIR="$RECORD_PACKAGE_DIR",RUN_TAG="$RUN_TAG",TRAIN_RATIO="$TRAIN_RATIO",SCHEDULE_RATIO="$SCHEDULE_RATIO",SNAPSHOT_RATIOS="$SNAPSHOT_RATIOS",TERMINAL_CLAMP_FRAC="$TERMINAL_CLAMP_FRAC",SNAPSHOT_K="$SNAPSHOT_K",SNAPSHOT_SPACING_FRAC="$SNAPSHOT_SPACING_FRAC",RECIPES_FILE="$RECIPES_FILE",SCREEN_SPLIT_TOKENS="$SCREEN_SPLIT_TOKENS",CANONICAL_SPLIT_TOKENS="$CANONICAL_SPLIT_TOKENS",WANDB_RUN="$WANDB_RUN",RECORD_LOCAL_ROOT="${RECORD_LOCAL_ROOT:-}",BRIDGES2_MODULES="${BRIDGES2_MODULES:-}" \
  "$PACKAGE_DIR/slurm/bridges2_explore.sbatch")
echo "explore_job=$JOB"
echo "run_tag=$RUN_TAG"
echo "monitor: squeue -j $JOB"
