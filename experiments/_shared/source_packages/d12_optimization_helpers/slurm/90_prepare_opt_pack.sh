#!/bin/bash
#SBATCH --job-name=opt_pack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${REPO:?REPO must be exported}/slurm/_skip_env.sh"
ROOT="${ROOT:?ROOT must be exported}"
PACK="${PACK:-${ROOT}/pack.pt}"
mkdir -p "${ROOT}"

# Match the shared-prefix learning rate to the branch baseline. Earlier suites
# trained a default-LR prefix and then branched at LR x0.30, which is not a
# clean optimizer comparison. All base LR families are scaled here.
PREFIX_LR_SCALE="${PREFIX_LR_SCALE:-1.0}"
scale_lr() {
  python - "$1" "$PREFIX_LR_SCALE" <<'PYLR'
import sys
print(float(sys.argv[1]) * float(sys.argv[2]))
PYLR
}
EMBEDDING_LR_SCALED=$(scale_lr "${BASE_EMBEDDING_LR:-0.2}")
UNEMBEDDING_LR_SCALED=$(scale_lr "${BASE_UNEMBEDDING_LR:-0.004}")
MATRIX_LR_SCALED=$(scale_lr "${BASE_MATRIX_LR:-0.02}")
SCALAR_LR_SCALED=$(scale_lr "${BASE_SCALAR_LR:-0.5}")
MATRIX_ADAM_LR_SCALED=$(scale_lr "${BASE_MATRIX_ADAM_LR:-0.003}")
echo "prefix LR scale=${PREFIX_LR_SCALE} embedding=${EMBEDDING_LR_SCALED} matrix=${MATRIX_LR_SCALED}"
python -m nanochat_meta.filter_suite prepare \
  --pack "${PACK}" \
  --depth "${DEPTH:-6}" \
  --aspect-ratio "${ASPECT_RATIO:-64}" \
  --head-dim "${HEAD_DIM:-128}" \
  --window-pattern "${WINDOW_PATTERN:-SSSL}" \
  --max-seq-len "${MAX_SEQ_LEN:-512}" \
  --device-batch-size "${DEVICE_BATCH_SIZE:-4}" \
  --grad-accum "${GRAD_ACCUM:-16}" \
  --optimizer "${OPTIMIZER:-native}" \
  --embedding-lr "${EMBEDDING_LR_SCALED}" \
  --unembedding-lr "${UNEMBEDDING_LR_SCALED}" \
  --matrix-lr "${MATRIX_LR_SCALED}" \
  --scalar-lr "${SCALAR_LR_SCALED}" \
  --matrix-adam-lr "${MATRIX_ADAM_LR_SCALED}" \
  --matrix-beta1 "${MATRIX_BETA1:-0.9}" \
  --matrix-beta2 "${MATRIX_BETA2:-0.95}" \
  --matrix-adam-weight-decay "${MATRIX_WD:-0.1}" \
  --prefix-steps "${PREFIX_STEPS:-1800}" \
  --total-iterations "${TOTAL_ITERATIONS:-5000}" \
  --warmup-steps "${WARMUP_STEPS:-40}" \
  --warmdown-ratio "${PACK_WARMDOWN_RATIO:-0}" \
  --final-lr-frac "${PACK_FINAL_LR_FRAC:-0.05}" \
  --history "${HISTORY:-64}" \
  --history-device "${HISTORY_DEVICE:-cuda}" \
  --history-dtype "${HISTORY_DTYPE:-bfloat16}" \
  --token-dim 8 --coeff-hidden 16 --coord-hidden 8 --coord-order 2 --coord-sample 1024 \
  --predictor-lr 0.001 --predictor-inner-steps 1 \
  --max-virtual-steps "${STREAM_EQUIV:-4000}" \
  --eval-batches "${EVAL_BATCHES:-128}" \
  --prefix-eval-every "${PREFIX_EVAL_EVERY:-200}" \
  --lawa-window 2 \
  --oracle-reference-accum 0 \
  --seed "${SEED:-1337}" \
  --skip-prefix-predictor-training
echo "PACK=${PACK}"
