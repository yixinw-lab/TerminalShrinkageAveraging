#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUITE_DIR
export WORK_ROOT="${WORK_ROOT:-/work/nvme/bhji/$USER/ExtrapProj/d22_delta_replication_v1}"
export NANOCHAT_SRC="${NANOCHAT_SRC:-$WORK_ROOT/nanochat}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$WORK_ROOT/nanochat_base}"
export RESULT_ROOT="${RESULT_ROOT:-$WORK_ROOT/results}"
ACCOUNT="${ACCOUNT_OVERRIDE:-bhji-dtai-gh}"
QOS="${QOS_OVERRIDE:-$ACCOUNT}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
[[ -f "$WORK_ROOT/EXPERIMENT_FROZEN.txt" ]] || { echo "Run prepare_d22_delta_checkout.sh first"; exit 2; }
mkdir -p "$RESULT_ROOT/logs"
submit() {
  local dep="$1"; shift
  local extra=()
  [[ -n "$dep" ]] && extra+=(--dependency="$dep")
  sbatch --parsable --account="$ACCOUNT" --qos="$QOS" --export=ALL,SUITE_DIR="$SUITE_DIR",WORK_ROOT="$WORK_ROOT",NANOCHAT_SRC="$NANOCHAT_SRC",NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",RESULT_ROOT="$RESULT_ROOT" "${extra[@]}" "$@"
}
setup=$(submit '' "$SUITE_DIR/slurm/00_setup.sh")
smoke=$(submit "afterok:$setup" "$SUITE_DIR/slurm/01_smoke.sh")
train=$(submit "afterok:$smoke" --array="0-5%$MAX_PARALLEL" "$SUITE_DIR/slurm/02_train_array.sh")
merge=$(submit "afterok:$train" "$SUITE_DIR/slurm/03_merge.sh")
cat > "$RESULT_ROOT/SUBMISSION.txt" <<EOF
suite=d22_delta_replication_suite_v1
submitted=$(date -Is)
work_root=$WORK_ROOT
nanochat_src=$NANOCHAT_SRC
result_root=$RESULT_ROOT
account=$ACCOUNT
qos=$QOS
max_parallel=$MAX_PARALLEL
setup_job=$setup
smoke_job=$smoke
train_array_job=$train
merge_job=$merge
EOF
cat "$RESULT_ROOT/SUBMISSION.txt"
echo
printf 'Submitted: setup=%s smoke=%s train=%s merge=%s\n' "$setup" "$smoke" "$train" "$merge"
