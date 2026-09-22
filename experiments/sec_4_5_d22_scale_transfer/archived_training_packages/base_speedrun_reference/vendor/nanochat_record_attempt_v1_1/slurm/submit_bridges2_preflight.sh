#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/logs"
EXPLORE_ENV=${1:-"$PACKAGE_DIR/config/explore.env"}
SITE_ENV=${2:-"$PACKAGE_DIR/config/site_bridges2.env"}
source "$EXPLORE_ENV"; source "$SITE_ENV"
: "${BRIDGES2_ACCOUNT:?}"
JOB=$(sbatch --parsable --account="$BRIDGES2_ACCOUNT" --partition="${BRIDGES2_PARTITION:-GPU}" \
  --export=ALL,REPO="$REPO",NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",RECORD_PACKAGE_DIR="$RECORD_PACKAGE_DIR",BRIDGES2_MODULES="${BRIDGES2_MODULES:-}" \
  "$PACKAGE_DIR/slurm/bridges2_preflight.sbatch")
echo "preflight_job=$JOB"
