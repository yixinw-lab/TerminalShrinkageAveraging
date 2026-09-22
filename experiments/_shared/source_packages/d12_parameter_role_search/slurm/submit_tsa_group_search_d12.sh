#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"
ROOT="${GROUP_ROOT:?set GROUP_ROOT to a NEW result directory}"
PACK_NATIVE="${PACK_NATIVE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
PACK_ADAMW="${PACK_ADAMW:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/pure_adamw_seed1337/pack.pt}"
DEV_SOURCE="$REPO/configs/tsa_group_search_dev_manifest.csv"
CONFIRM_SOURCE="$REPO/configs/tsa_group_search_confirm_manifest.csv"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
GROUP_NAME="${GROUP_NAME:-d12_tsa_group_search_v1}"

for f in \
  "$SUB" "$PACK_NATIVE" "$PACK_ADAMW" "$DEV_SOURCE" "$CONFIRM_SOURCE" \
  "$REPO/nanochat_meta/structured_tsa_d12.py" \
  "$REPO/nanochat_meta/tsa_group_search_d12.py" \
  "$REPO/nanochat_meta/paper_main_d12.py" \
  "$REPO/nanochat_meta/filter_suite.py" \
  "$REPO/nanochat_meta/skip_suite.py" \
  "$REPO/tools/analyze_tsa_group_search_d12.py" \
  "$D/_tsa_group_search_env.sh" \
  "$D/90_tsa_group_search_smoke.sh" \
  "$D/91_tsa_group_search_dev_array.sh" \
  "$D/92_tsa_group_search_freeze.sh" \
  "$D/93_tsa_group_search_confirm_array.sh" \
  "$D/94_tsa_group_search_merge.sh"; do
  [[ -e "$f" ]] || { echo "missing required file: $f" >&2; exit 2; }
done
[[ -x "$SUB" ]] || { echo "$SUB is not executable" >&2; exit 2; }
if find "$ROOT" -name result.json -print -quit 2>/dev/null | grep -q .; then
  echo "GROUP_ROOT already contains completed runs; use a new root: $ROOT" >&2
  exit 2
fi

mkdir -p "$REPO/logs" "$ROOT"
cp "$DEV_SOURCE" "$ROOT/DEV_MANIFEST.csv"
cp "$CONFIRM_SOURCE" "$ROOT/CONFIRM_MANIFEST.csv"
DEV_MANIFEST="$ROOT/DEV_MANIFEST.csv"
CONFIRM_MANIFEST="$ROOT/CONFIRM_MANIFEST.csv"
DEV_N=$(( $(wc -l < "$DEV_MANIFEST") - 1 ))
CONFIRM_N=$(( $(wc -l < "$CONFIRM_MANIFEST") - 1 ))
[[ "$DEV_N" -eq 8 ]] || { echo "expected 8 dev rows, got $DEV_N" >&2; exit 2; }
[[ "$CONFIRM_N" -eq 16 ]] || { echo "expected 16 confirm rows, got $CONFIRM_N" >&2; exit 2; }

{
  echo "suite=d12_tsa_group_search_v1"
  echo "created_at=$(date -Is)"
  echo "repo=$REPO"
  echo "root=$ROOT"
  echo "group_name=$GROUP_NAME"
  echo "dev_runs=$DEV_N"
  echo "confirm_runs=$CONFIRM_N"
  echo "max_parallel=$MAX_PARALLEL"
  echo "native_pack=$PACK_NATIVE"
  echo "pure_adamw_pack=$PACK_ADAMW"
  echo "design=8 native development streams -> frozen family winners -> 8 native + 8 paired pure-AdamW confirmations"
  echo "primary_contrast=development-selected overall structured rule minus frozen scalar on fresh native 96-batch holdout"
  echo "secondary_contrast=overall structured rule minus development-selected hidden/rest winner"
  echo "snapshot_policy=K=8,s=32,float32 CPU snapshots; 180G host memory requested"
} > "$ROOT/SUBMISSION.txt"

(
  cd "$REPO"
  sha256sum \
    nanochat_meta/tsa_group_search_d12.py \
    tools/analyze_tsa_group_search_d12.py \
    configs/tsa_group_search_dev_manifest.csv \
    configs/tsa_group_search_confirm_manifest.csv \
    slurm/_tsa_group_search_env.sh \
    slurm/90_tsa_group_search_smoke.sh \
    slurm/91_tsa_group_search_dev_array.sh \
    slurm/92_tsa_group_search_freeze.sh \
    slurm/93_tsa_group_search_confirm_array.sh \
    slurm/94_tsa_group_search_merge.sh \
    slurm/submit_tsa_group_search_d12.sh
) > "$ROOT/SOURCE_MANIFEST.sha256"

COMMON="ALL,REPO=$REPO,ROOT=$ROOT,DEV_MANIFEST=$DEV_MANIFEST,CONFIRM_MANIFEST=$CONFIRM_MANIFEST,PACK_NATIVE=$PACK_NATIVE,PACK_ADAMW=$PACK_ADAMW"
SMOKE=$("$SUB" --parsable --export="$COMMON" "$D/90_tsa_group_search_smoke.sh")
echo "smoke_job=$SMOKE" | tee -a "$ROOT/SUBMISSION.txt"
DEV=$("$SUB" --parsable --dependency="afterok:$SMOKE" --array="0-$((DEV_N-1))%$MAX_PARALLEL" --export="$COMMON" "$D/91_tsa_group_search_dev_array.sh")
echo "dev_array_job=$DEV" | tee -a "$ROOT/SUBMISSION.txt"
FREEZE=$("$SUB" --parsable --dependency="afterok:$DEV" --export="$COMMON" "$D/92_tsa_group_search_freeze.sh")
echo "freeze_job=$FREEZE" | tee -a "$ROOT/SUBMISSION.txt"
FROZEN_RULES="$ROOT/FROZEN_GROUP_RULES.json"
COMMON_CONFIRM="$COMMON,FROZEN_RULES=$FROZEN_RULES"
CONFIRM=$("$SUB" --parsable --dependency="afterok:$FREEZE" --array="0-$((CONFIRM_N-1))%$MAX_PARALLEL" --export="$COMMON_CONFIRM" "$D/93_tsa_group_search_confirm_array.sh")
echo "confirm_array_job=$CONFIRM" | tee -a "$ROOT/SUBMISSION.txt"
MERGE=$("$SUB" --parsable --dependency="afterok:$CONFIRM" --export="$COMMON_CONFIRM" "$D/94_tsa_group_search_merge.sh")
echo "merge_job=$MERGE" | tee -a "$ROOT/SUBMISSION.txt"

cat <<TXT
Submitted D12 TSA group-search suite.
  root:       $ROOT
  smoke:      $SMOKE
  dev:        $DEV   ($DEV_N native runs)
  freeze:     $FREEZE
  confirm:    $CONFIRM   ($CONFIRM_N runs: 8 native + 8 pure AdamW)
  merge:      $MERGE

All five stages request one GPU because DeltaAI rejects GPU-partition jobs with no GPU request.
Training stages request 180G host memory and retain only 8 full float32 snapshots.
TXT
