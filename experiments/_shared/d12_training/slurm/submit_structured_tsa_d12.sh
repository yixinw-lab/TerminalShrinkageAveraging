#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"
GROUP_NAME="${GROUP_NAME:-d12_structured_tsa_v1}"
ROOT="${GROUP_ROOT:-$REPO/outputs/optimization_paper/$GROUP_NAME}"
PACK_NATIVE="${PACK_NATIVE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
PACK_ADAMW="${PACK_ADAMW:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/pure_adamw_seed1337/pack.pt}"
DEV_MANIFEST="${DEV_MANIFEST:-$REPO/configs/structured_tsa_dev_manifest.csv}"
CONFIRM_MANIFEST="${CONFIRM_MANIFEST:-$REPO/configs/structured_tsa_confirm_manifest.csv}"
DEV_MAX_PARALLEL="${DEV_MAX_PARALLEL:-10}"
CONFIRM_MAX_PARALLEL="${CONFIRM_MAX_PARALLEL:-30}"
EVAL_MAX_PARALLEL="${EVAL_MAX_PARALLEL:-30}"

for f in \
  "$SUB" \
  "$PACK_NATIVE" \
  "$PACK_ADAMW" \
  "$DEV_MANIFEST" \
  "$CONFIRM_MANIFEST" \
  "$REPO/nanochat_meta/structured_tsa_d12.py" \
  "$REPO/tools/freeze_structured_tsa_rules.py" \
  "$REPO/tools/analyze_structured_tsa_d12.py" \
  "$D/90_structured_tsa_smoke.sh" \
  "$D/91_structured_tsa_dev_array.sh" \
  "$D/92_structured_tsa_confirm_train_array.sh" \
  "$D/93_structured_tsa_freeze.sh" \
  "$D/94_structured_tsa_confirm_eval_array.sh" \
  "$D/95_structured_tsa_merge.sh"; do
  [[ -e "$f" ]] || { echo "missing required file: $f" >&2; exit 2; }
done
[[ -x "$SUB" ]] || { echo "$SUB is not executable" >&2; exit 2; }

DEV_N=$(( $(wc -l < "$DEV_MANIFEST") - 1 ))
CONFIRM_N=$(( $(wc -l < "$CONFIRM_MANIFEST") - 1 ))
[[ "$DEV_N" -eq 10 ]] || { echo "expected 10 development rows, found $DEV_N" >&2; exit 2; }
[[ "$CONFIRM_N" -eq 30 ]] || { echo "expected 30 confirmation rows, found $CONFIRM_N" >&2; exit 2; }

if [[ -e "$ROOT" && -n "$(find "$ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && "${ALLOW_EXISTING:-0}" != 1 ]]; then
  echo "Refusing to mix results into nonempty root: $ROOT" >&2
  echo "Choose a new GROUP_NAME or set ALLOW_EXISTING=1 only when resuming this exact suite." >&2
  exit 2
fi
mkdir -p "$ROOT" "$REPO/logs"
cp "$DEV_MANIFEST" "$ROOT/DEV_MANIFEST.csv"
cp "$CONFIRM_MANIFEST" "$ROOT/CONFIRM_MANIFEST.csv"

RESOLVED_ROOT=$(readlink -f "$ROOT" 2>/dev/null || printf '%s' "$ROOT")
case "$RESOLVED_ROOT" in
  /u/*) echo "WARNING: result root resolves under /u. DeltaAI job I/O is better placed under /work/nvme or /work/hdd." >&2 ;;
esac

{
  echo "suite=d12_structured_tsa_suite_v1"
  echo "created_at=$(date -Is)"
  echo "repo=$REPO"
  echo "resolved_root=$RESOLVED_ROOT"
  echo "group_name=$GROUP_NAME"
  echo "native_pack=$(readlink -f "$PACK_NATIVE" 2>/dev/null || printf '%s' "$PACK_NATIVE")"
  echo "pure_adamw_pack=$(readlink -f "$PACK_ADAMW" 2>/dev/null || printf '%s' "$PACK_ADAMW")"
  echo "development_design=2 optimizers x 5 streams at 10pct floor; structured 9x9 and scalar 0.05 calibration grids"
  echo "confirmation_design=2 optimizers x 3 floors x 5 paired streams"
  echo "primary_estimand=native 10pct floor: selected-scalar BPB minus selected-structured BPB"
  echo "selection_data=development trajectories and alpha-selection validation block only"
  echo "confirmation_data=separate streams and untouched holdout validation block"
  echo "checkpoint_rule=K8 spacing32"
  echo "model_seed=1337"
  echo "development_jobs=$DEV_N"
  echo "confirmation_training_jobs=$CONFIRM_N"
  echo "confirmation_evaluation_jobs=$CONFIRM_N"
  echo "dev_max_parallel=$DEV_MAX_PARALLEL"
  echo "confirm_max_parallel=$CONFIRM_MAX_PARALLEL"
  echo "eval_max_parallel=$EVAL_MAX_PARALLEL"
  if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    echo "git_head=$(git -C "$REPO" rev-parse HEAD)"
  else
    echo "git_head=unavailable"
  fi
} > "$ROOT/SUBMISSION.txt"
if git -C "$REPO" status --short >/dev/null 2>&1; then
  git -C "$REPO" status --short > "$ROOT/GIT_STATUS.txt"
fi
(
  cd "$REPO"
  sha256sum \
    nanochat_meta/structured_tsa_d12.py \
    tools/freeze_structured_tsa_rules.py \
    tools/analyze_structured_tsa_d12.py \
    configs/structured_tsa_dev_manifest.csv \
    configs/structured_tsa_confirm_manifest.csv \
    slurm/90_structured_tsa_smoke.sh \
    slurm/91_structured_tsa_dev_array.sh \
    slurm/92_structured_tsa_confirm_train_array.sh \
    slurm/93_structured_tsa_freeze.sh \
    slurm/94_structured_tsa_confirm_eval_array.sh \
    slurm/95_structured_tsa_merge.sh \
    slurm/submit_structured_tsa_d12.sh
) > "$ROOT/SOURCE_MANIFEST.sha256"

COMMON="ALL,REPO=$REPO,ROOT=$ROOT,PACK_NATIVE=$PACK_NATIVE,PACK_ADAMW=$PACK_ADAMW,DEV_MANIFEST=$ROOT/DEV_MANIFEST.csv,CONFIRM_MANIFEST=$ROOT/CONFIRM_MANIFEST.csv"
SMOKE=$("$SUB" --parsable --export="$COMMON" "$D/90_structured_tsa_smoke.sh")
echo "smoke_job=$SMOKE" | tee -a "$ROOT/SUBMISSION.txt"

DEV=$("$SUB" --parsable \
  --dependency="afterok:$SMOKE" \
  --array="0-$((DEV_N-1))%$DEV_MAX_PARALLEL" \
  --export="$COMMON" \
  "$D/91_structured_tsa_dev_array.sh")
echo "development_array_job=$DEV" | tee -a "$ROOT/SUBMISSION.txt"

CONFIRM=$("$SUB" --parsable \
  --dependency="afterok:$SMOKE" \
  --array="0-$((CONFIRM_N-1))%$CONFIRM_MAX_PARALLEL" \
  --export="$COMMON" \
  "$D/92_structured_tsa_confirm_train_array.sh")
echo "confirmation_training_array_job=$CONFIRM" | tee -a "$ROOT/SUBMISSION.txt"

FREEZE=$("$SUB" --parsable \
  --dependency="afterok:$DEV" \
  --export="$COMMON" \
  "$D/93_structured_tsa_freeze.sh")
echo "freeze_job=$FREEZE" | tee -a "$ROOT/SUBMISSION.txt"

EVAL=$("$SUB" --parsable \
  --dependency="afterok:$CONFIRM:$FREEZE" \
  --array="0-$((CONFIRM_N-1))%$EVAL_MAX_PARALLEL" \
  --export="$COMMON" \
  "$D/94_structured_tsa_confirm_eval_array.sh")
echo "confirmation_evaluation_array_job=$EVAL" | tee -a "$ROOT/SUBMISSION.txt"

MERGE=$("$SUB" --parsable \
  --dependency="afterok:$EVAL" \
  --export="$COMMON" \
  "$D/95_structured_tsa_merge.sh")
echo "merge_job=$MERGE" | tee -a "$ROOT/SUBMISSION.txt"

cat <<EOF
Submitted D12 structured-TSA suite.
  root:                $ROOT
  smoke:               $SMOKE
  development array:   $DEV
  confirmation train:  $CONFIRM
  freeze:              $FREEZE
  confirmation eval:   $EVAL
  merge:               $MERGE

Development and confirmation training launch in parallel after the smoke test.
The confirmation evaluation cannot start until FROZEN_RULES.json has been produced.
EOF
