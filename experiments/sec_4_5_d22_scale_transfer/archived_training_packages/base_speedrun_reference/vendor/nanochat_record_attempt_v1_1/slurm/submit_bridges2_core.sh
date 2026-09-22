#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/logs"
TOP=${1:?usage: submit_bridges2_core.sh /path/to/top_candidates.csv [site.env]}
SITE_ENV=${2:-"$PACKAGE_DIR/config/site_bridges2.env"}
EXPLORE_ENV=${EXPLORE_ENV:-"$PACKAGE_DIR/config/explore.env"}
source "$SITE_ENV"; source "$EXPLORE_ENV"
: "${BRIDGES2_ACCOUNT:?}"
N=$(python3 - "$TOP" <<'PY'
import csv,sys
print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))
PY
)
(( N > 0 )) || { echo "no candidates" >&2; exit 1; }
CORE_DIR=$(cd "$(dirname "$TOP")/.." && pwd)/core_results
JOB=$(sbatch --parsable \
  --account="$BRIDGES2_ACCOUNT" \
  --partition="${BRIDGES2_PARTITION:-GPU}" \
  --time="${BRIDGES2_TIME_CORE:-04:00:00}" \
  --array="0-$((N-1))%${CORE_PARALLEL:-2}" \
  --export=ALL,REPO="$REPO",NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",RECORD_PACKAGE_DIR="$RECORD_PACKAGE_DIR",TOP_CANDIDATES="$(readlink -f "$TOP")",CORE_OUTPUT_DIR="$CORE_DIR",CANONICAL_SPLIT_TOKENS="$CANONICAL_SPLIT_TOKENS",BRIDGES2_MODULES="${BRIDGES2_MODULES:-}" \
  "$PACKAGE_DIR/slurm/bridges2_core_ladder.sbatch")
echo "core_array_job=$JOB"
echo "results=$CORE_DIR"
echo "after completion: python $PACKAGE_DIR/record_tools/freeze_candidate.py --results-dir $CORE_DIR --output $PACKAGE_DIR/config/frozen_candidate.env --threshold $CORE_THRESHOLD --safety-margin $CORE_SAFETY_MARGIN"
