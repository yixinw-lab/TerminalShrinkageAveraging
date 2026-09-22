#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_COMMIT=e09bc164162f35da0b5b8315be791e9e974a4c3a
WORK_ROOT="${WORK_ROOT:-/work/nvme/bhji/$USER/ExtrapProj/d22_delta_replication_v1}"
NANOCHAT_SRC="${NANOCHAT_SRC:-$WORK_ROOT/nanochat}"
mkdir -p "$WORK_ROOT"
if [[ ! -d "$NANOCHAT_SRC/.git" ]]; then
  echo "Cloning NanoChat into $NANOCHAT_SRC"
  git clone https://github.com/karpathy/nanochat.git "$NANOCHAT_SRC"
fi
git -C "$NANOCHAT_SRC" fetch origin "$BASE_COMMIT" || true
git -C "$NANOCHAT_SRC" checkout --detach "$BASE_COMMIT"
HEAD=$(git -C "$NANOCHAT_SRC" rev-parse HEAD)
[[ "$HEAD" == "$BASE_COMMIT" ]] || { echo "wrong HEAD: $HEAD"; exit 1; }
if [[ -n "$(git -C "$NANOCHAT_SRC" status --porcelain)" ]]; then
  echo "ERROR: exact checkout is dirty before instrumentation" >&2; git -C "$NANOCHAT_SRC" status --short; exit 1
fi
python3 "$SUITE_DIR/tools/patch_base_train_d22.py" "$NANOCHAT_SRC" | tee "$WORK_ROOT/PATCH_MANIFEST.txt"
python3 -m py_compile "$NANOCHAT_SRC/scripts/base_train_d22_delta.py"
cp "$SUITE_DIR/configs/d22_delta_manifest.csv" "$WORK_ROOT/MANIFEST.csv"
sha256sum "$NANOCHAT_SRC/scripts/base_train.py" "$NANOCHAT_SRC/scripts/base_train_d22_delta.py" > "$WORK_ROOT/CODE_SHA256.txt"
cat > "$WORK_ROOT/EXPERIMENT_FROZEN.txt" <<EOF
suite=d22_delta_replication_suite_v1
base_commit=$BASE_COMMIT
prepared_at=$(date -Is)
repo=$NANOCHAT_SRC
schedule_calibration_ratio=9.4
physical_stop_step=10172
control_floor=none
treatment_floor=0.15
snapshot_steps=9381,9494,9607,9720,9833,9946,10059,10172
posthoc_alphas=0,0.55,0.587,1
posthoc_ewa_beta=0.75
seeds=42,43,44
primary_estimator=tsa_alpha_0.587
primary_estimand=paired schedule-by-estimator interaction (raw schedule effect minus TSA(.587) schedule effect)
hardware_note=DeltaAI two-node 8-GPU runs are for quality/uncertainty, not official leaderboard timing
EOF
sha256sum "$WORK_ROOT/EXPERIMENT_FROZEN.txt" "$WORK_ROOT/MANIFEST.csv" >> "$WORK_ROOT/CODE_SHA256.txt"
echo "PREPARED=$WORK_ROOT"
echo "NANOCHAT_SRC=$NANOCHAT_SRC"
