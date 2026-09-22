#!/bin/bash
# Cluster submission wrapper.
#
# Old usage still works:
#   ./sub.sh 20_oracle.sh
#
# It also accepts sbatch options before the script:
#   ./sub.sh --parsable --dependency=afterok:123 job.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

# Optional per-campaign charging allocation.
# env.sh remains the default.
ACCOUNT="${ACCOUNT_OVERRIDE:-${ACCOUNT}}"
PARTITION="${PARTITION_OVERRIDE:-${PARTITION}}"
QOS="${QOS_OVERRIDE:-${ACCOUNT}}"

SBATCH_ARGS=()

while [[ $# -gt 0 && "$1" == --* ]]; do
    SBATCH_ARGS+=("$1")
    shift
done

if [[ $# -lt 1 ]]; then
    echo "usage: $0 [sbatch options ...] SCRIPT [script args ...]" >&2
    exit 2
fi

SCRIPT="$1"
shift

exec sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    "${SBATCH_ARGS[@]}" \
    "${SCRIPT}" "$@"
