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
RUN_TAG=${RUN_TAG:-baseline_r${SLUG}_$(date +%Y%m%d_%H%M%S)}
MODEL_TAG="${RUN_TAG}_train"
RESULT_DIR="$ROOT/results/$RUN_TAG"
mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/train.log"
exec > >(tee -a "$LOG") 2>&1
cat > "$RESULT_DIR/run_config.txt" <<CFG
arm=baseline_x
target_ratio=$RATIO
schedule_ratio=$RATIO
terminal_clamp=disabled
output=raw
commit=$(git -C "$REPO" rev-parse HEAD)
CFG

echo "===== DIRECT BASELINE: ratio $RATIO schedule calibrated to $RATIO, no clamp, raw ====="
cd "$REPO"
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=22 --target-param-data-ratio="$RATIO" \
  --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=999999 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --model-tag="$MODEL_TAG" --run=dummy
STEP=$(final_step_for_tag "$MODEL_TAG")
echo "final_step=$STEP" >> "$RESULT_DIR/run_config.txt"
copy_final_meta "$MODEL_TAG" "$STEP" "$RESULT_DIR"
run_canonical_eval "$MODEL_TAG" "$STEP" "$RATIO" raw "$RESULT_DIR/raw_core.json" "$RESULT_DIR/raw_core.log"
python "$SUITE_DIR/tools/summarize_arm.py" --arm baseline_x --result-dir "$RESULT_DIR" --threshold "$CORE_THRESHOLD"
cleanup_checkpoint_tag "$MODEL_TAG"
bash "$SUITE_DIR/tools/package_results.sh" "$RESULT_DIR" "$ROOT/results/${RUN_TAG}_results.tar.gz"
echo "RESULT=$RESULT_DIR"
