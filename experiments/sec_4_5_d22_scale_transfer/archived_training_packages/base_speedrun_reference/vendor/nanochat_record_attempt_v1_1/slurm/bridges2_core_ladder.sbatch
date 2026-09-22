#!/usr/bin/env bash
#SBATCH --job-name=nc_record_core
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h100-80:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/nc_record_core_%A_%a.out
#SBATCH --error=logs/nc_record_core_%A_%a.err
set -euo pipefail
: "${REPO:?}"
: "${NANOCHAT_BASE_DIR:?}"
: "${RECORD_PACKAGE_DIR:?}"
: "${TOP_CANDIDATES:?}"
: "${CORE_OUTPUT_DIR:?}"
: "${CANONICAL_SPLIT_TOKENS:=20971520}"

cd "$REPO"
[[ -z "${BRIDGES2_MODULES:-}" ]] || module load ${BRIDGES2_MODULES}
source .venv/bin/activate
export NANOCHAT_BASE_DIR OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$CORE_OUTPUT_DIR"

mapfile -t FIELDS < <(python - "$TOP_CANDIDATES" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
r=rows[int(sys.argv[2])]
for k in ('model_tag','endpoint_step','ratio','recipe'):
    print(r[k])
PY
)
MODEL_TAG=${FIELDS[0]}
STEP=${FIELDS[1]}
RATIO=${FIELDS[2]}
RECIPE=${FIELDS[3]}
SAFE=$(echo "$MODEL_TAG" | tr -c 'A-Za-z0-9_.-' '_')
python "$RECORD_PACKAGE_DIR/record_tools/eval_candidate.py" \
  --repo "$REPO" --model-tag "$MODEL_TAG" --step "$STEP" \
  --ratio "$RATIO" --recipe "$RECIPE" --nproc 8 \
  --split-tokens "$CANONICAL_SPLIT_TOKENS" --device-batch-size 32 --max-per-task -1 \
  --output "$CORE_OUTPUT_DIR/${SAFE}.json" \
  --log "$CORE_OUTPUT_DIR/${SAFE}.log"
