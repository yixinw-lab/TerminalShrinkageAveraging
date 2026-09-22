#!/usr/bin/env bash
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=05:00:00
#SBATCH --job-name=d22dtrain
#SBATCH --output=d22dtrain_%A_%a.log
set -euo pipefail
SUITE_DIR="${SUITE_DIR:?set SUITE_DIR}"
source "$SUITE_DIR/slurm/_d22_env.sh"
MANIFEST="$WORK_ROOT/MANIFEST.csv"
row=$(awk -F, -v i="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==i {print; exit}' "$MANIFEST")
[[ -n "$row" ]] || { echo "manifest row missing for $SLURM_ARRAY_TASK_ID"; exit 2; }
IFS=, read -r idx arm floor seed tag <<< "$row"
RUN_DIR="$RESULT_ROOT/runs/$tag"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/RUN_FROZEN.txt" <<EOF
index=$idx
arm=$arm
floor=$floor
experiment_seed=$seed
model_tag=$tag
schedule_ratio=9.4
physical_stop=10172
snapshot_steps=9381,9494,9607,9720,9833,9946,10059,10172
EOF
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
port=$((29500 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID) % 1000))
echo "arm=$arm seed=$seed floor=$floor nodes=$SLURM_JOB_NODELIST head=$head_node:$port"
srun torchrun \
  --nnodes="$SLURM_NNODES" --nproc_per_node="$SLURM_GPUS_PER_NODE" \
  --rdzv_id="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --rdzv_backend=c10d --rdzv_endpoint="${head_ip}:${port}" \
  -m scripts.base_train_d22_delta -- \
  --depth=22 --target-param-data-ratio=9.4 --device-batch-size=32 --total-batch-size=524288 \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy \
  --model-tag="$tag" --experiment-seed="$seed" --stop-after-step=10172 \
  --terminal-lr-floor="$floor" \
  --snapshot-steps=9381,9494,9607,9720,9833,9946,10059,10172 \
  --snapshot-dir="$RUN_DIR" --skip-standard-final-checkpoint \
  --posthoc-eval --posthoc-eval-tokens=$((80*524288)) \
  --posthoc-alphas=0,0.55,0.587,1 --posthoc-ewa-beta=0.75 \
  2>&1 | tee "$RUN_DIR/torchrun.log"
[[ -s "$RUN_DIR/metrics.json" ]] || { echo "missing metrics.json"; exit 3; }
python - "$RUN_DIR/metrics.json" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p));
assert d['world_size']==8, d
assert d['physical_stop_step']==10172, d
assert d['schedule_horizon_steps']>10172, d
for k in ['raw','tsa_alpha_0.55','tsa_alpha_0.587','tsa_alpha_1','ewa_beta_0.75']:
    assert k in d['metrics_bpb'], (k,d)
print(json.dumps(d,indent=2,sort_keys=True))
PY
sha256sum "$RUN_DIR"/model_step*.pt "$RUN_DIR/metrics.json" > "$RUN_DIR/SHA256.txt"
echo "RUN PASS $tag"
