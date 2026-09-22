#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_repo
need_8_gpus
RATIO=${1:-$DEFAULT_CANDIDATE_RATIO}
python - "$RATIO" <<'PY'
import sys
x=float(sys.argv[1])
assert 7.0 <= x <= 10.0, x
PY
SLUG=${RATIO/./p}
RUN_TAG=${RUN_TAG:-tsa_clamp_r${SLUG}_$(date +%Y%m%d_%H%M%S)}
MODEL_TAG="${RUN_TAG}_train"
RESULT_DIR="$ROOT/results/$RUN_TAG"
SNAP_DIR="$ROOT/snapshot_scratch/$RUN_TAG"
mkdir -p "$RESULT_DIR" "$SNAP_DIR"
LOG="$RESULT_DIR/train.log"
exec > >(tee -a "$LOG") 2>&1
cat > "$RESULT_DIR/run_config.txt" <<CFG
arm=tsa_clamp_x
target_ratio=$RATIO
schedule_ratio=$RATIO
terminal_clamp=$TERMINAL_CLAMP_FRAC
tsa_alpha=$TSA_ALPHA
snapshot_k=$SNAPSHOT_K
snapshot_spacing_frac=$SNAPSHOT_SPACING_FRAC
commit=$(git -C "$REPO" rev-parse HEAD)
CFG

echo "===== TREATMENT: ratio $RATIO schedule calibrated to $RATIO, clamp $TERMINAL_CLAMP_FRAC, TSA $TSA_ALPHA ====="
cd "$REPO"
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=22 --target-param-data-ratio="$RATIO" \
  --record-schedule-param-data-ratio="$RATIO" \
  --record-terminal-lr-clamp-frac="$TERMINAL_CLAMP_FRAC" \
  --record-snapshot-ratios="$RATIO" \
  --record-snapshot-k="$SNAPSHOT_K" \
  --record-snapshot-spacing-frac="$SNAPSHOT_SPACING_FRAC" \
  --record-snapshot-dir="$SNAP_DIR" \
  --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=999999 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --model-tag="$MODEL_TAG" --run=dummy
STEP=$(final_step_for_tag "$MODEL_TAG")
echo "final_step=$STEP" >> "$RESULT_DIR/run_config.txt"
copy_final_meta "$MODEL_TAG" "$STEP" "$RESULT_DIR"
cp "$SNAP_DIR/manifest.json" "$RESULT_DIR/snapshot_manifest.json"
python "$SUITE_DIR/tools/validate_treatment_manifest.py" --manifest "$RESULT_DIR/snapshot_manifest.json" --ratio "$RATIO" --clamp "$TERMINAL_CLAMP_FRAC" --k "$SNAPSHOT_K"

python "$PACKAGE_DIR/record_tools/merge_record_snapshots.py" \
  --snapshot-dir "$SNAP_DIR" \
  --source-checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" \
  --base-checkpoints "$NANOCHAT_BASE_DIR/base_checkpoints" \
  --output-prefix "$RUN_TAG" \
  --recipes "blend:$TSA_ALPHA" \
  --output-manifest "$RESULT_DIR/candidate.csv"
TSA_TAG=$(python - "$RESULT_DIR/candidate.csv" <<'PY'
import csv,sys
r=next(csv.DictReader(open(sys.argv[1])))
print(r['model_tag'])
PY
)
TSA_STEP=$(python - "$RESULT_DIR/candidate.csv" <<'PY'
import csv,sys
r=next(csv.DictReader(open(sys.argv[1])))
print(r['endpoint_step'])
PY
)
[[ "$TSA_STEP" == "$STEP" ]] || { echo "ERROR: TSA step $TSA_STEP != raw step $STEP" >&2; exit 4; }

echo "tsa_model_tag=$TSA_TAG" >> "$RESULT_DIR/run_config.txt"
# Snapshots and optimizer state are no longer needed after the TSA checkpoint exists.
rm -rf "$SNAP_DIR"
find "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" -maxdepth 1 -type f -name 'optim_*' -delete 2>/dev/null || true

echo "===== EVALUATING TREATMENT RAW OUTPUT ====="
run_canonical_eval "$MODEL_TAG" "$STEP" "$RATIO" raw "$RESULT_DIR/raw_core.json" "$RESULT_DIR/raw_core.log"
echo "===== EVALUATING TSA OUTPUT ====="
run_canonical_eval "$TSA_TAG" "$STEP" "$RATIO" "blend:$TSA_ALPHA" "$RESULT_DIR/tsa_core.json" "$RESULT_DIR/tsa_core.log"
python "$SUITE_DIR/tools/summarize_arm.py" --arm tsa_clamp_x --result-dir "$RESULT_DIR" --threshold "$CORE_THRESHOLD"
cleanup_checkpoint_tag "$MODEL_TAG"
cleanup_checkpoint_tag "$TSA_TAG"
bash "$SUITE_DIR/tools/package_results.sh" "$RESULT_DIR" "$ROOT/results/${RUN_TAG}_results.tar.gz"
echo "RESULT=$RESULT_DIR"
