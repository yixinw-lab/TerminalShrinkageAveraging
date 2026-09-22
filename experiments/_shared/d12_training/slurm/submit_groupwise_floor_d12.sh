#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$D/.." && pwd)}"
SUB="$REPO/sub.sh"
ROOT="${GROUP_ROOT:?set GROUP_ROOT to a NEW result directory}"
PACK_NATIVE="${PACK_NATIVE:-$REPO/outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt}"
MANIFEST_SOURCE="$REPO/configs/groupwise_floor_manifest.csv"
PROTOCOL_SOURCE="$REPO/configs/groupwise_floor_protocol.json"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
GROUP_NAME="${GROUP_NAME:-d12_tsa_groupwise_floor_v1}"

for f in \
  "$SUB" "$PACK_NATIVE" "$MANIFEST_SOURCE" "$PROTOCOL_SOURCE" \
  "$REPO/nanochat_meta/structured_tsa_d12.py" \
  "$REPO/nanochat_meta/groupwise_floor_d12.py" \
  "$REPO/nanochat_meta/paper_main_d12.py" \
  "$REPO/nanochat_meta/filter_suite.py" \
  "$REPO/nanochat_meta/skip_suite.py" \
  "$REPO/tools/analyze_groupwise_floor_d12.py" \
  "$D/_groupwise_floor_env.sh" \
  "$D/90_groupwise_floor_smoke.sh" \
  "$D/91_groupwise_floor_array.sh" \
  "$D/92_groupwise_floor_merge.sh"; do
  [[ -e "$f" ]] || { echo "missing required file: $f" >&2; exit 2; }
done
[[ -x "$SUB" ]] || { echo "$SUB is not executable" >&2; exit 2; }
if find "$ROOT" -name result.json -print -quit 2>/dev/null | grep -q .; then
  echo "GROUP_ROOT already contains completed runs; use a new root: $ROOT" >&2
  exit 2
fi

mkdir -p "$REPO/logs" "$ROOT"
cp "$MANIFEST_SOURCE" "$ROOT/RUN_MANIFEST.csv"
cp "$PROTOCOL_SOURCE" "$ROOT/PROTOCOL.json"
RUN_MANIFEST="$ROOT/RUN_MANIFEST.csv"
PROTOCOL="$ROOT/PROTOCOL.json"
N=$(( $(wc -l < "$RUN_MANIFEST") - 1 ))
[[ "$N" -eq 32 ]] || { echo "expected 32 run rows, got $N" >&2; exit 2; }

{
  echo "suite=d12_tsa_groupwise_floor_v1"
  echo "created_at=$(date -Is)"
  echo "repo=$REPO"
  echo "root=$ROOT"
  echo "group_name=$GROUP_NAME"
  echo "run_count=$N"
  echo "paired_streams=8"
  echo "arms=uniform05,uniform10,matched_uniform,mapped"
  echo "max_parallel=$MAX_PARALLEL"
  echo "native_pack=$PACK_NATIVE"
  PROTOCOL_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol_sha256"])' "$PROTOCOL")
  echo "protocol_sha256=$PROTOCOL_SHA"
  echo "selection_policy=none; all floor rules, estimators, streams, and contrasts frozen before training"
  echo "primary=structured(mapped) vs structured(parameter-count-matched uniform floor)"
  echo "primary_interaction=mapped-vs-matched schedule effect under raw minus same effect under structured TSA"
  echo "frozen_tsa=embedding:0.00 hidden:0.65 unembed:0.45 rest:0.25"
  echo "mapped_floor_formula=rho=0.05+0.10*alpha/0.65"
  echo "snapshot_policy=K=8,s=32,float32 CPU snapshots; 180G host memory"
} > "$ROOT/SUBMISSION.txt"

(
  cd "$REPO"
  sha256sum \
    nanochat_meta/groupwise_floor_d12.py \
    tools/analyze_groupwise_floor_d12.py \
    configs/groupwise_floor_manifest.csv \
    configs/groupwise_floor_protocol.json \
    slurm/_groupwise_floor_env.sh \
    slurm/90_groupwise_floor_smoke.sh \
    slurm/91_groupwise_floor_array.sh \
    slurm/92_groupwise_floor_merge.sh \
    slurm/submit_groupwise_floor_d12.sh
) > "$ROOT/SOURCE_MANIFEST.sha256"

COMMON="ALL,REPO=$REPO,ROOT=$ROOT,RUN_MANIFEST=$RUN_MANIFEST,PROTOCOL=$PROTOCOL,PACK_NATIVE=$PACK_NATIVE"
SMOKE=$("$SUB" --parsable --export="$COMMON" "$D/90_groupwise_floor_smoke.sh")
echo "smoke_job=$SMOKE" | tee -a "$ROOT/SUBMISSION.txt"
RUN=$("$SUB" --parsable --dependency="afterok:$SMOKE" --array="0-$((N-1))%$MAX_PARALLEL" --export="$COMMON" "$D/91_groupwise_floor_array.sh")
echo "run_array_job=$RUN" | tee -a "$ROOT/SUBMISSION.txt"
MERGE=$("$SUB" --parsable --dependency="afterok:$RUN" --export="$COMMON" "$D/92_groupwise_floor_merge.sh")
echo "merge_job=$MERGE" | tee -a "$ROOT/SUBMISSION.txt"

cat <<TXT
Submitted D12 predeclared groupwise-floor suite.
  root:    $ROOT
  smoke:   $SMOKE
  runs:    $RUN   (32 jobs = 8 paired streams x 4 arms)
  merge:   $MERGE

No development/freeze stage exists in this suite: the protocol is already frozen.
All jobs request one GPU because DeltaAI's GPU partition rejects GPU-less jobs.
Training jobs request 180G host memory and save K=8 float32 CPU snapshots.
TXT
