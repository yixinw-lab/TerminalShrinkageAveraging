#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PACKAGE_DIR"
EXPLORE_ENV=${1:-"$PACKAGE_DIR/config/explore.env"}
SITE_ENV=${2:-"$PACKAGE_DIR/config/site_deltaai.env"}
source "$EXPLORE_ENV"; source "$SITE_ENV"
: "${DELTA_ACCOUNT:?set DELTA_ACCOUNT}"
mkdir -p "$PACKAGE_DIR/logs"
ARGS=(--parsable --account="$DELTA_ACCOUNT" --partition="${DELTA_PARTITION:-ghx4}" --time="${DELTA_TIME:-05:00:00}")
[[ -z "${DELTA_QOS:-}" ]] || ARGS+=(--qos="$DELTA_QOS")
JOB=$(sbatch "${ARGS[@]}" \
  --export=ALL,REPO="$REPO",NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",RECORD_PACKAGE_DIR="$RECORD_PACKAGE_DIR",RUN_TAG="$RUN_TAG",TRAIN_RATIO="$TRAIN_RATIO",SCHEDULE_RATIO="$SCHEDULE_RATIO",SNAPSHOT_RATIOS="${DELTA_SNAPSHOT_RATIOS:-9.2,9.4}",TERMINAL_CLAMP_FRAC="$TERMINAL_CLAMP_FRAC",SNAPSHOT_K="$SNAPSHOT_K",SNAPSHOT_SPACING_FRAC="$SNAPSHOT_SPACING_FRAC",RECIPES_FILE="$RECIPES_FILE",SCREEN_SPLIT_TOKENS="${DELTA_SCREEN_SPLIT_TOKENS:-2097152}",WANDB_RUN="$WANDB_RUN",RECORD_LOCAL_ROOT="${RECORD_LOCAL_ROOT:-}" \
  "$PACKAGE_DIR/slurm/deltaai_probe.sbatch")
echo "deltaai_probe_job=$JOB"
echo "NOTE: this is not official leaderboard timing; DeltaAI is 4-GPU GH200 per node."
