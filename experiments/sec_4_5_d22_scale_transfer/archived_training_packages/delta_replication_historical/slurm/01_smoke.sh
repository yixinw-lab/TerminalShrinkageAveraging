#!/usr/bin/env bash
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --job-name=d22dsmoke
#SBATCH --output=d22dsmoke_%j.log
set -euo pipefail
SUITE_DIR="${SUITE_DIR:?set SUITE_DIR}"
source "$SUITE_DIR/slurm/_d22_env.sh"
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
port=$((29500 + SLURM_JOB_ID % 1000))
mkdir -p "$RESULT_ROOT/smoke"
srun torchrun \
  --nnodes="$SLURM_NNODES" --nproc_per_node="$SLURM_GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="${head_ip}:${port}" \
  -m scripts.base_train_d22_delta -- \
  --depth=22 --target-param-data-ratio=9.4 --device-batch-size=32 --total-batch-size=524288 \
  --fp8 --liger-cross-entropy --learnable-rmsnorm \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy \
  --model-tag=d22_delta_smoke --experiment-seed=42 --stop-after-step=20 \
  --terminal-lr-floor=-1 --skip-standard-final-checkpoint \
  2>&1 | tee "$RESULT_ROOT/smoke/torchrun.log"
grep -q 'Distributed world size: 8' "$RESULT_ROOT/smoke/torchrun.log"
grep -q 'Physical stop step: 20' "$RESULT_ROOT/smoke/torchrun.log"
grep -q 'Total training time:' "$RESULT_ROOT/smoke/torchrun.log"
{
 echo "SMOKE PASS"
 echo "completed=$(date -Is)"
 echo "nodes=$SLURM_JOB_NODELIST"
 grep -E 'GPU:|Distributed world size:|Calculated number of iterations|Physical stop step:' "$RESULT_ROOT/smoke/torchrun.log" | head -20
} > "$RESULT_ROOT/SMOKE.txt"
cat "$RESULT_ROOT/SMOKE.txt"
