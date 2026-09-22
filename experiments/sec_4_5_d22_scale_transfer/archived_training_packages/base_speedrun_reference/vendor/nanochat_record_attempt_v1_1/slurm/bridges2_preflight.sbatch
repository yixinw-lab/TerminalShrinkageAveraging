#!/usr/bin/env bash
#SBATCH --job-name=nc_record_preflight
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h100-80:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/nc_record_preflight_%j.out
#SBATCH --error=logs/nc_record_preflight_%j.err
set -euo pipefail
: "${REPO:?}"; : "${NANOCHAT_BASE_DIR:?}"; : "${RECORD_PACKAGE_DIR:?}"
cd "$REPO"
[[ -z "${BRIDGES2_MODULES:-}" ]] || module load ${BRIDGES2_MODULES}
source .venv/bin/activate
export NANOCHAT_BASE_DIR OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
EXPECTED_GPUS=8 "$RECORD_PACKAGE_DIR/tools/check_environment.sh" "$REPO"
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=22 --num-iterations=250 --target-param-data-ratio=9.4 \
  --record-schedule-param-data-ratio=9.4 --record-terminal-lr-clamp-frac=0.15 \
  --device-batch-size=32 --total-batch-size=524288 \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --model-tag="record_preflight_${SLURM_JOB_ID}" --run=dummy
rm -rf "$NANOCHAT_BASE_DIR/base_checkpoints/record_preflight_${SLURM_JOB_ID}"
