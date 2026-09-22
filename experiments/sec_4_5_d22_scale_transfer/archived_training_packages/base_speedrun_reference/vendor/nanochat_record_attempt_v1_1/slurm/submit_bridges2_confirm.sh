#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/logs"
CONFIRM_ENV=${1:-"$PACKAGE_DIR/config/confirm.env"}
SITE_ENV=${2:-"$PACKAGE_DIR/config/site_bridges2.env"}
[[ -f "$CONFIRM_ENV" ]] || { echo "missing $CONFIRM_ENV" >&2; exit 1; }
source "$CONFIRM_ENV"; source "$SITE_ENV"
: "${BRIDGES2_ACCOUNT:?}"; : "${FROZEN_ENV:?}"; [[ -f "$FROZEN_ENV" ]]
N=${N_REPEATS:-6}
JOB=$(sbatch --parsable \
  --account="$BRIDGES2_ACCOUNT" \
  --partition="${BRIDGES2_PARTITION:-GPU}" \
  --time="${BRIDGES2_TIME_CONFIRM:-05:00:00}" \
  --array="0-$((N-1))%${CONFIRM_PARALLEL:-2}" \
  --export=ALL,REPO="$REPO",NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",RECORD_PACKAGE_DIR="$RECORD_PACKAGE_DIR",FROZEN_ENV="$(readlink -f "$FROZEN_ENV")",WANDB_PREFIX="${WANDB_PREFIX:-record_confirm}",CANONICAL_SPLIT_TOKENS="${CANONICAL_SPLIT_TOKENS:-20971520}",RECORD_LOCAL_ROOT="${RECORD_LOCAL_ROOT:-}",BRIDGES2_MODULES="${BRIDGES2_MODULES:-}" \
  "$PACKAGE_DIR/slurm/bridges2_confirm_array.sbatch")
echo "confirm_array_job=$JOB"
echo "after completion locate: $NANOCHAT_BASE_DIR/record_attempts/confirm_$JOB"
echo "summarize with: python $PACKAGE_DIR/record_tools/summarize_confirmations.py --results-dir $NANOCHAT_BASE_DIR/record_attempts/confirm_$JOB --output $NANOCHAT_BASE_DIR/record_attempts/confirm_$JOB/DIGEST.txt"
