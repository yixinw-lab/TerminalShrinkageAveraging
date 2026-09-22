#!/usr/bin/env bash
# Unattended RunPod supervisor v2 for the missing D22 direct-9.0 + 15% floor + TSA row.
# Intended to be launched from macOS with RUNPOD_API_KEY already exported.
# It waits for an 8x H100 80GB HBM3 pod, then runs seeds 42,43,44 for one missing cell:
#   direct ratio-9.0 schedule + 15% terminal floor + TSA(alpha=0.587).
# The exact physical checkpoint window is held fixed at the paper's 113-step spacing.
# Every returned fixed-alpha model gets canonical validation BPB and full DCLM CORE.
# A post-hoc alpha grid is BPB-screened descriptively; it does not alter the primary row.
# It downloads an auditable evidence archive and deletes the pod on normal completion.

set -u

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set. Export it in this shell before launching.}"

DOWNLOADS="${DOWNLOADS:-$HOME/Downloads}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="${PKG:-$BUNDLE_DIR/runpod_d22_causal_speedrun_v2.tar.gz}"
EXPECTED_SHA="501afed8dacbaaf93117d3efe6e54d2836adca9423fdee0bc1b70646a37d9bb1"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE COMMUNITY}"
INTERVAL="${INTERVAL:-30}"
RUN_NAME="${RUN_NAME:-d22-missing-direct9-floor15-s42-s44-$(date +%Y%m%d-%H%M%S)}"
STATE_DIR="${STATE_DIR:-$DOWNLOADS/d22_runpod_missing_row_results/${RUN_NAME}}"
SSH_KEY="$STATE_DIR/d22_missing_row_ed25519"
LOCAL_SUMMARY="$STATE_DIR/FINAL_SUMMARY.txt"
LOCAL_EVIDENCE="$STATE_DIR/evidence.tar.gz"
LOCAL_FRESH="$STATE_DIR/FRESH_RESULTS.csv"
LOCAL_ALL="$STATE_DIR/N3_ALL_RUNS.csv"
LOCAL_N3="$STATE_DIR/N3_SUMMARY.csv"
LOCAL_LATEX="$STATE_DIR/N3_TABLE.tex"

mkdir -p "$STATE_DIR"
mkdir -p "$DOWNLOADS/d22_runpod_missing_row_results"
ln -sfn "$STATE_DIR" "$DOWNLOADS/d22_runpod_missing_row_latest"
printf '%s\n' "$$" > "$STATE_DIR/supervisor.pid"
printf '%s\n' "$RUN_NAME" > "$STATE_DIR/run_name.txt"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

notify() {
  local msg="$1"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${msg//\"/\\\"}\" with title \"RunPod D22 missing-row supervisor\" sound name \"Glass\"" >/dev/null 2>&1 || true
  fi
}

# The launcher performs the cross-supervisor exclusion check *before* starting caffeinate.
# Do not repeat a broad `pgrep -f ...missing_row_supervisor...` here: once launched, that
# pattern matches this supervisor's own caffeinate wrapper and caused v1 to self-abort.
# We still guard against the older *different* D22 waiters if this script is invoked directly.
OTHER_WAITERS="$(pgrep -fl 'wait_for_8xh100\.sh|runpod_supervisor_d22_paired_v1\.py|runpod_d22_virtual_horizon_supervisor|runpod_d22_n3_supervisor' 2>/dev/null || true)"
if [[ -n "$OTHER_WAITERS" ]]; then
  log "ERROR: another RunPod D22 waiter/supervisor is still running:"
  printf '%s\n' "$OTHER_WAITERS"
  log "Stop the older waiter first, then launch this supervisor."
  exit 2
fi

if [[ ! -f "$PKG" ]]; then
  log "ERROR: package not found: $PKG"
  exit 2
fi

ACTUAL_SHA="$(shasum -a 256 "$PKG" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  log "ERROR: package SHA mismatch."
  log "Expected: $EXPECTED_SHA"
  log "Actual:   $ACTUAL_SHA"
  exit 2
fi
log "Verified package SHA256: $ACTUAL_SHA"

# Dedicated ephemeral SSH key so automation never has to guess which account key has a private half locally.
if [[ ! -f "$SSH_KEY" ]]; then
  ssh-keygen -q -t ed25519 -N '' -f "$SSH_KEY" -C "d22-missing-row-${RUN_NAME}"
fi
SSH_FINGERPRINT="$(ssh-keygen -lf "$SSH_KEY.pub" -E sha256 | awk '{print $2}')"
log "Registering ephemeral SSH key: $SSH_FINGERPRINT"
ADD_KEY_OUT="$(runpodctl ssh add-key --key-file "$SSH_KEY.pub" 2>&1 || true)"
printf '%s\n' "$ADD_KEY_OUT" > "$STATE_DIR/add_ssh_key.out"
# If the exact key already exists, add-key may report an error; list-keys is the source of truth.
if ! runpodctl ssh list-keys 2>/dev/null | grep -Fq "$SSH_FINGERPRINT"; then
  log "ERROR: could not verify ephemeral SSH key registration. See $STATE_DIR/add_ssh_key.out"
  exit 3
fi

POD_ID=""
SSH_TARGET=""
SSH_PORT=""
WORKER_STARTED=0
CLEANED=0

cleanup_key() {
  runpodctl ssh remove-key --fingerprint "$SSH_FINGERPRINT" >/dev/null 2>&1 || true
}

on_interrupt() {
  log "Supervisor interrupted."
  if [[ -n "$POD_ID" ]]; then
    log "Stopping pod $POD_ID to cap GPU billing while preserving its disk."
    runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
    log "Pod was stopped, not deleted. State/evidence may still be recoverable manually."
  fi
  cleanup_key
  exit 130
}
trap on_interrupt INT TERM

extract_pod_id_by_name() {
  runpodctl pod list --name "$RUN_NAME" -o json 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
if isinstance(d, dict):
    if isinstance(d.get("pods"), list): d=d["pods"]
    else: d=[d]
if not isinstance(d, list) or not d: sys.exit(1)
for x in d:
    if isinstance(x, dict) and x.get("id"):
        print(x["id"]); sys.exit(0)
sys.exit(1)
' 2>/dev/null
}

log "Waiting for one 8x NVIDIA H100 80GB HBM3 pod on any of: $CLOUD_TYPES."
log "Run name: $RUN_NAME"
log "Retry interval after a full Secure+Community cycle: ${INTERVAL}s"
log "No public IP is required; automation uses RunPod SSH, which keeps the eligible worker pool larger."
log "Once acquired, GPU billing starts automatically."

attempt=0
while [[ -z "$POD_ID" ]]; do
  EXISTING_ID="$(extract_pod_id_by_name || true)"
  if [[ -n "$EXISTING_ID" ]]; then
    POD_ID="$EXISTING_ID"
    log "Found existing pod with this run name: $POD_ID"
    break
  fi

  for TRY_CLOUD in $CLOUD_TYPES; do
    attempt=$((attempt + 1))
    # Absolute timestamps work with the current CLI/backend even though duration strings such as 4h do not.
    STOP_AT="$(date -u -v+10H '+%Y-%m-%dT%H:%M:%SZ')"
    log "Capacity attempt $attempt on $TRY_CLOUD (server-side GPU stop failsafe: $STOP_AT)"

    # Deliberately do not request --public-ip or an exposed 22/tcp port. SSH is enabled by default,
    # and avoiding those placement requirements makes more Community workers eligible.
    CREATE_OUT="$(runpodctl pod create \
      --name "$RUN_NAME" \
      --template-id runpod-torch-v280 \
      --gpu-id 'NVIDIA H100 80GB HBM3' \
      --gpu-count 8 \
      --cloud-type "$TRY_CLOUD" \
      --container-disk-in-gb 50 \
      --volume-in-gb 250 \
      --volume-mount-path /workspace \
      --ssh \
      --stop-after "$STOP_AT" 2>&1)"
    CREATE_RC=$?
    printf '%s\n' "$CREATE_OUT"

    # A successful create can take a few seconds to appear in pod list. Never try the other
    # cloud immediately after a success, because that could create a duplicate expensive pod.
    if [[ $CREATE_RC -eq 0 ]]; then
      for _ in 1 2 3 4 5; do
        POD_ID="$(extract_pod_id_by_name || true)"
        [[ -n "$POD_ID" ]] && break
        sleep 2
      done
      if [[ -n "$POD_ID" ]]; then
        log "POD ACQUIRED on $TRY_CLOUD: $POD_ID"
        notify "8x H100 pod acquired on $TRY_CLOUD; experiment is starting"
        break 2
      fi
      log "ERROR: create reported success on $TRY_CLOUD but pod ID did not become visible; stopping to avoid a duplicate."
      cleanup_key
      exit 4
    fi

    POD_ID="$(extract_pod_id_by_name || true)"
    if [[ -n "$POD_ID" ]]; then
      log "POD ACQUIRED on $TRY_CLOUD despite nonzero client return: $POD_ID"
      notify "8x H100 pod acquired on $TRY_CLOUD; experiment is starting"
      break 2
    fi

    if echo "$CREATE_OUT" | grep -qiE 'no longer any instances available|no instances available|requested specifications'; then
      # Capacity miss: immediately try the other cloud before sleeping.
      continue
    fi

    if echo "$CREATE_OUT" | grep -qiE 'rate.?limit|too many requests|429'; then
      log "RunPod rate limit encountered; sleeping 60s before continuing."
      sleep 60
      continue
    fi

    log "ERROR: non-capacity pod-create failure on $TRY_CLOUD; refusing to retry blindly."
    cleanup_key
    exit 4
  done

  [[ -n "$POD_ID" ]] && break
  sleep "$INTERVAL"
done

printf '%s\n' "$POD_ID" > "$STATE_DIR/pod_id.txt"

# Wait for RunPod to publish a real SSH-over-TCP endpoint and for the pod to accept commands.
# runpodctl 2.x emits JSON here. Parse JSON directly; do not scrape the human ssh command.
# RunPod's own golden paths recommend deleting/recreating a bad machine draw if runtime/SSH
# never becomes ready after about 5-6 minutes.
ssh_ready_for_current_pod() {
  local ssh_attempt SSH_INFO PARSED IP PORT SSH_CMD
  log "Waiting for SSH/runtime readiness on pod $POD_ID..."

  for ssh_attempt in $(seq 1 24); do
    # Keep both diagnostic sources for postmortem.
    runpodctl pod get "$POD_ID" -o json > "$STATE_DIR/pod_get_latest.json" 2>&1 || true
    SSH_INFO="$(runpodctl ssh info "$POD_ID" 2>&1 || true)"
    printf '%s\n' "$SSH_INFO" > "$STATE_DIR/ssh_info.txt"

    PARSED="$(printf '%s\n' "$SSH_INFO" | python3 -c '
import json,sys,re,shlex
s=sys.stdin.read().strip()
try:
    d=json.loads(s)
except Exception:
    sys.exit(1)
if not isinstance(d,dict): sys.exit(1)
# Current runpodctl uses ip/port/ssh_key/ssh_command. Older builds may use sshCommand.
ip=d.get("ip") or d.get("host") or ""
port=d.get("port") or ""
cmd=d.get("ssh_command") or d.get("sshCommand") or ""
if (not ip or not port) and cmd:
    try: toks=shlex.split(cmd)
    except Exception: toks=cmd.split()
    for t in toks:
        if "@" in t and not t.startswith("-"):
            ip=t.split("@",1)[1]
            break
    for i,t in enumerate(toks[:-1]):
        if t=="-p": port=toks[i+1]
if not ip or not port: sys.exit(1)
print(ip)
print(port)
' 2>/dev/null || true)"

    IP="$(printf '%s\n' "$PARSED" | sed -n '1p')"
    PORT="$(printf '%s\n' "$PARSED" | sed -n '2p')"

    if [[ -n "$IP" && -n "$PORT" ]]; then
      SSH_TARGET="root@$IP"
      SSH_PORT="$PORT"
      SSH_BASE=(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$SSH_PORT" "$SSH_TARGET")
      if "${SSH_BASE[@]}" 'echo SSH_READY' 2>/dev/null | grep -q SSH_READY; then
        log "SSH ready: $SSH_TARGET port $SSH_PORT (attempt $ssh_attempt)"
        return 0
      fi
    fi

    if (( ssh_attempt % 4 == 0 )); then
      log "Still waiting for runtime/SSH (${ssh_attempt}/24; about $((ssh_attempt*15/60)) min)."
    fi
    sleep 15
  done
  return 1
}

# Bad-draw loop: if a newly rented machine never produces a usable runtime/SSH endpoint,
# delete it (no experiment state exists yet) and resume the capacity hunt instead of burning GPUs.
while ! ssh_ready_for_current_pod; do
  log "WARNING: pod $POD_ID never became SSH-ready within ~6 minutes. Treating as a bad machine draw."
  log "Deleting unusable pod immediately and resuming the 8xH100 capacity hunt."
  runpodctl pod delete "$POD_ID" >/dev/null 2>&1 || runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
  printf '%s\n' "$POD_ID" >> "$STATE_DIR/bad_draw_pod_ids.txt"
  POD_ID=""
  SSH_TARGET=""
  SSH_PORT=""

  # Re-enter the same Secure+Community capacity loop.
  while [[ -z "$POD_ID" ]]; do
    for TRY_CLOUD in $CLOUD_TYPES; do
      attempt=$((attempt + 1))
      STOP_AT="$(date -u -v+10H '+%Y-%m-%dT%H:%M:%SZ')"
      log "Capacity attempt $attempt on $TRY_CLOUD after bad draw (failsafe: $STOP_AT)"
      CREATE_OUT="$(runpodctl pod create \
        --name "$RUN_NAME" \
        --template-id runpod-torch-v280 \
        --gpu-id 'NVIDIA H100 80GB HBM3' \
        --gpu-count 8 \
        --cloud-type "$TRY_CLOUD" \
        --container-disk-in-gb 50 \
        --volume-in-gb 250 \
        --volume-mount-path /workspace \
        --ssh \
        --stop-after "$STOP_AT" 2>&1)"
      CREATE_RC=$?
      printf '%s\n' "$CREATE_OUT"
      if [[ $CREATE_RC -eq 0 ]]; then
        for _ in 1 2 3 4 5; do
          POD_ID="$(extract_pod_id_by_name || true)"
          [[ -n "$POD_ID" ]] && break
          sleep 2
        done
        if [[ -n "$POD_ID" ]]; then
          log "POD RE-ACQUIRED on $TRY_CLOUD: $POD_ID"
          printf '%s\n' "$POD_ID" > "$STATE_DIR/pod_id.txt"
          notify "Replacement 8x H100 pod acquired on $TRY_CLOUD"
          break 2
        fi
      fi
      POD_ID="$(extract_pod_id_by_name || true)"
      if [[ -n "$POD_ID" ]]; then
        log "POD RE-ACQUIRED despite nonzero create response: $POD_ID"
        printf '%s\n' "$POD_ID" > "$STATE_DIR/pod_id.txt"
        break 2
      fi
      if echo "$CREATE_OUT" | grep -qiE 'no longer any instances available|no instances available|requested specifications'; then
        continue
      fi
      log "ERROR: non-capacity create failure while replacing a bad draw."
      cleanup_key
      exit 5
    done
    [[ -n "$POD_ID" ]] || sleep "$INTERVAL"
  done
done

# Upload through ordinary SSH stdin. This works even when the RunPod basic SSH proxy does not support SCP/SFTP.
log "Uploading verified experiment archive..."
if ! cat "$PKG" | "${SSH_BASE[@]}" 'cat > /workspace/runpod_d22_causal_speedrun_v2.tar.gz'; then
  log "ERROR: package upload failed."
  runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
  cleanup_key
  exit 6
fi

REMOTE_SHA="$("${SSH_BASE[@]}" 'sha256sum /workspace/runpod_d22_causal_speedrun_v2.tar.gz | cut -d" " -f1' 2>/dev/null || true)"
if [[ "$REMOTE_SHA" != "$EXPECTED_SHA" ]]; then
  log "ERROR: remote package SHA mismatch: $REMOTE_SHA"
  runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
  cleanup_key
  exit 6
fi
log "Remote package SHA verified."

# Build the remote worker locally so it is preserved with the run evidence.
REMOTE_WORKER_LOCAL="$STATE_DIR/n3_remote_worker.sh"
cat > "$REMOTE_WORKER_LOCAL" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

THRESHOLD="0.256525"
SUITE="/workspace/runpod_d22_causal_speedrun_v2"
ROOT="/workspace/nanochat-causal-speedrun-v2"
STATUS="/workspace/n3_status.txt"
WORKLOG="/workspace/n3_remote_worker.log"
SUMMARY="/workspace/FINAL_SUMMARY.txt"
DONE="/workspace/n3_worker.done"
EVIDENCE="/workspace/d22_n3_evidence.tar.gz"
COMPACT="/workspace/n3_compact"
FRESH="/workspace/FRESH_RESULTS.csv"
ALL="/workspace/N3_ALL_RUNS.csv"
N3="/workspace/N3_SUMMARY.csv"
LATEX="/workspace/N3_TABLE.tex"
ALPHA_ALL="/workspace/ALPHA_SCREEN_ALL.csv"
ALPHA_SUM="/workspace/ALPHA_SCREEN_SUMMARY.csv"
ALPHA_REC="/workspace/ALPHA_RECOMMENDATION.txt"
SEEDS=(42 43 44)

mkdir -p "$COMPACT" /workspace/n3_logs /workspace/model_evidence
exec >>"$WORKLOG" 2>&1

status() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$STATUS"
}

json_field() {
  python3 - "$1" "$2" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); v=d[sys.argv[2]]
except Exception:
    sys.exit(1)
print(v)
PY
}

find_core_json() {
  local result_dir="$1" wanted="$2"
  python3 - "$result_dir" "$wanted" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); wanted=sys.argv[2]
for p in sorted(root.glob('*.json')):
    try:d=json.loads(p.read_text())
    except Exception:continue
    if isinstance(d,dict) and 'core_metric' in d and str(d.get('recipe',''))==wanted:
        print(p);sys.exit(0)
sys.exit(1)
PY
}

copy_meta_and_hash_model() {
  local label="$1" result_json="$2"
  python3 - "$label" "$result_json" /workspace/model_evidence <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
label,jpath,outdir=sys.argv[1:4]
d=json.loads(Path(jpath).read_text()); out=Path(outdir); out.mkdir(exist_ok=True)
meta=Path(d['meta'])
rec={k:d.get(k) for k in ('model_tag','step','ratio','recipe','val_bpb','core_metric','training_minutes')}
rec['label']=label;rec['meta']=str(meta)
if meta.is_file():
    shutil.copy2(meta,out/f'{label}_{meta.name}')
    if meta.name.startswith('meta_') and meta.name.endswith('.json'):
        model=meta.with_name('model_'+meta.name[5:-5]+'.pt')
        rec['model_path']=str(model)
        if model.is_file():
            h=hashlib.sha256()
            with model.open('rb') as f:
                for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
            rec['model_sha256']=h.hexdigest()
(out/f'{label}_record.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
PY
}

cleanup_checkpoint_dir_from_json() {
  python3 - "$1" <<'PY'
import json,shutil,sys
from pathlib import Path
try:d=json.load(open(sys.argv[1])); meta=Path(d['meta']).resolve()
except Exception:sys.exit(0)
root=Path('/workspace/nanochat-causal-speedrun-v2/data/base_checkpoints').resolve()
p=meta.parent
if p.is_dir() and root in p.parents: shutil.rmtree(p,ignore_errors=True)
PY
}

make_evidence() {
  status "Packaging compact evidence archive"
  cd /workspace
  rm -f "$EVIDENCE" /workspace/n3_evidence_filelist.txt
  {
    printf '%s\n' \
      n3_status.txt n3_remote_worker.log FINAL_SUMMARY.txt n3_worker.done \
      FRESH_RESULTS.csv N3_ALL_RUNS.csv N3_SUMMARY.csv N3_TABLE.tex \
      ALPHA_SCREEN_ALL.csv ALPHA_SCREEN_SUMMARY.csv ALPHA_RECOMMENDATION.txt \
      nvidia-smi-L.txt nvidia-smi-topo.txt gpu-query.csv disk_initial.txt disk_final.txt \
      verify_suite.log experiment_patch.diff seed_patch.diff manifest_validator_preflight.txt
    find n3_logs n3_compact model_evidence -type f -print 2>/dev/null || true
    for d in nanochat-causal-speedrun-v2/results/n3_*; do
      [[ -d "$d" ]] || continue
      find "$d" -type f \( -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.txt' -o -name '*_results.tar.gz' \) -print 2>/dev/null || true
    done
    if [[ -d "$SUITE" ]]; then
      printf '%s\n' \
        runpod_d22_causal_speedrun_v2/config/experiment.env \
        runpod_d22_causal_speedrun_v2/runpod/run_treatment_90_seeded_screen.sh \
        runpod_d22_causal_speedrun_v2/tools/validate_direct9_manifest.py
    fi
  } | awk 'NF && !seen[$0]++' > /workspace/n3_evidence_filelist.txt
  tar -czf "$EVIDENCE" --ignore-failed-read -T /workspace/n3_evidence_filelist.txt 2>/dev/null || true
  sha256sum "$EVIDENCE" > /workspace/d22_n3_evidence.sha256 2>/dev/null || true
}

finish() {
  local rc=$?
  set +e
  local preserve=0
  [[ $rc -ne 0 && -f /workspace/any_training_started.flag ]] && preserve=1
  status "Remote worker exiting with code $rc (preserve_pod=$preserve)"
  make_evidence
  printf 'exit_code=%s\nevidence=%s\npreserve_pod=%s\n' "$rc" "$EVIDENCE" "$preserve" > "$DONE"
  exit "$rc"
}
trap finish EXIT

: > "$STATUS"
status "PHASE 1/6: hardware verification"
nvidia-smi -L | tee /workspace/nvidia-smi-L.txt
GPU_COUNT="$(nvidia-smi -L | grep -c 'H100')"
[[ "$GPU_COUNT" -eq 8 ]] || { status "FATAL: expected exactly 8 H100 GPUs, found $GPU_COUNT"; exit 20; }
nvidia-smi topo -m | tee /workspace/nvidia-smi-topo.txt
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total --format=csv | tee /workspace/gpu-query.csv
df -h /workspace | tee /workspace/disk_initial.txt

status "PHASE 2/6: extract and pin the direct-9.0 treatment experiment"
cd /workspace
rm -rf "$SUITE"
tar --no-same-owner -xzf /workspace/runpod_d22_causal_speedrun_v2.tar.gz
cd "$SUITE"
if ! bash verify_suite.sh > /workspace/verify_suite.log 2>&1; then
  status "WARNING: internal verifier reported a packaged-bytecode mismatch; source/config assertions continue"
fi
cp config/experiment.env config/experiment.env.original
cp runpod/run_treatment_x.sh runpod/run_treatment_x.sh.original

python3 - <<'PY_PATCH'
from pathlib import Path
import re
cfg=Path('config/experiment.env'); s=cfg.read_text()
s,n=re.subn(r'(?m)^TSA_ALPHA=0\.60$', 'TSA_ALPHA=0.587', s); assert n==1,n
# Keep the physical eight-checkpoint window identical to the paper treatment:
# round(10172 * frac) = 113.
s,n=re.subn(r'(?m)^SNAPSHOT_SPACING_FRAC=.*$', 'SNAPSHOT_SPACING_FRAC=0.011108926464805349', s); assert n==1,n
cfg.write_text(s)

validator=Path('tools/validate_direct9_manifest.py')
validator.write_text("""#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--ratio',type=float,required=True);ap.add_argument('--clamp',type=float,required=True);ap.add_argument('--k',type=int,required=True);ap.add_argument('--spacing',type=int,required=True);a=ap.parse_args()
m=json.loads(Path(a.manifest).read_text());uc=m.get('user_config',{})
close=lambda x,y: math.isclose(float(x),float(y),rel_tol=0,abs_tol=1e-10)
assert m.get('complete') is True
assert close(uc.get('target_param_data_ratio'),a.ratio)
assert close(uc.get('record_schedule_param_data_ratio'),a.ratio)
assert close(uc.get('record_terminal_lr_clamp_frac'),a.clamp)
assert int(m['snapshot_k'])==a.k and int(m['snapshot_spacing_steps'])==a.spacing
assert int(m['train_num_iterations'])==10172 and int(m['schedule_num_iterations'])==10172
es=[e for e in m.get('endpoints',{}).values() if close(e.get('ratio'),a.ratio)];assert len(es)==1
endpoint=int(es[0]['endpoint_step']);assert endpoint==10172
expected=[endpoint-a.spacing*i for i in range(a.k-1,-1,-1)]
steps=[int(x) for x in es[0]['snapshot_steps']]
assert steps==expected and [int(x) for x in m.get('required_steps',[])]==expected
assert not m.get('missing_steps') and sorted(int(x) for x in m.get('captured',{}))==expected
print(f'direct9 manifest PASS ratio={a.ratio:g} endpoint={endpoint} spacing={a.spacing} steps={steps}')
""")
validator.chmod(0o755)

t=Path('runpod/run_treatment_x.sh').read_text()
needle='need_8_gpus\n'; assert t.count(needle)==1
t=t.replace(needle,needle+': "${EXPERIMENT_SEED:?EXPERIMENT_SEED is required}"\n',1)
t=t.replace('schedule_ratio=$RATIO\n','schedule_ratio=$RATIO\nexperiment_seed=$EXPERIMENT_SEED\n',1)
needle='  --model-tag="$MODEL_TAG" --run=dummy\n'; assert t.count(needle)==1
t=t.replace(needle,'  --model-tag="$MODEL_TAG" --experiment-seed="$EXPERIMENT_SEED" --run=dummy\n',1)
old='python "$SUITE_DIR/tools/validate_treatment_manifest.py" --manifest "$RESULT_DIR/snapshot_manifest.json" --ratio "$RATIO" --clamp "$TERMINAL_CLAMP_FRAC" --k "$SNAPSHOT_K"'
new='python "$SUITE_DIR/tools/validate_direct9_manifest.py" --manifest "$RESULT_DIR/snapshot_manifest.json" --ratio "$RATIO" --clamp "$TERMINAL_CLAMP_FRAC" --k "$SNAPSHOT_K" --spacing 113'
assert t.count(old)==1;t=t.replace(old,new,1)
old='  --recipes "blend:$TSA_ALPHA" \\\n  --output-manifest "$RESULT_DIR/candidate.csv"'
new='  --recipes "blend:0.30,blend:0.35,blend:0.40,blend:0.45,blend:0.50,blend:0.55,blend:0.587,blend:0.60,blend:0.65,blend:0.70,blend:0.75,blend:0.80,blend:1.00" \\\n  --output-manifest "$RESULT_DIR/candidate.csv"'
assert t.count(old)==1;t=t.replace(old,new,1)
old="""TSA_TAG=$(python - "$RESULT_DIR/candidate.csv" <<'PY'
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
)"""
new="""TSA_TAG=$(python - "$RESULT_DIR/candidate.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
r=next(x for x in rows if x['recipe']=='blend:0.587')
print(r['model_tag'])
PY
)
TSA_STEP=$(python - "$RESULT_DIR/candidate.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
r=next(x for x in rows if x['recipe']=='blend:0.587')
print(r['endpoint_step'])
PY
)"""
assert t.count(old)==1;t=t.replace(old,new,1)
needle='[[ "$TSA_STEP" == "$STEP" ]] || { echo "ERROR: TSA step $TSA_STEP != raw step $STEP" >&2; exit 4; }\n\n'
assert t.count(needle)==1
t=t.replace(needle,needle+'''echo "===== DESCRIPTIVE ALPHA GRID BPB SCREEN (does not select primary row) ====="\npython "$PACKAGE_DIR/record_tools/screen_candidates.py" \\\n  --repo "$REPO" --candidates "$RESULT_DIR/candidate.csv" --output-dir "$RESULT_DIR/alpha_screen" \\\n  --nproc 8 --split-tokens 4194304 --device-batch-size "$DEVICE_BATCH_SIZE"\n\npython - "$RESULT_DIR/candidate.csv" "$TSA_TAG" "$NANOCHAT_BASE_DIR/base_checkpoints" <<'PY'\nimport csv,shutil,sys\nfrom pathlib import Path\nrows=list(csv.DictReader(open(sys.argv[1]))); keep=sys.argv[2]; root=Path(sys.argv[3])\nfor r in rows:\n    tag=r['model_tag']\n    if tag != keep:\n        shutil.rmtree(root/tag,ignore_errors=True)\nPY\n\n''',1)
oldblock='''echo "===== EVALUATING TREATMENT RAW OUTPUT ====="\nrun_canonical_eval "$MODEL_TAG" "$STEP" "$RATIO" raw "$RESULT_DIR/raw_core.json" "$RESULT_DIR/raw_core.log"\necho "===== EVALUATING TSA OUTPUT ====="\nrun_canonical_eval "$TSA_TAG" "$STEP" "$RATIO" "blend:$TSA_ALPHA" "$RESULT_DIR/tsa_core.json" "$RESULT_DIR/tsa_core.log"'''
newblock='''echo "===== EVALUATING PRIMARY FIXED TSA(.587) OUTPUT ====="\nrun_canonical_eval "$TSA_TAG" "$STEP" "$RATIO" "blend:$TSA_ALPHA" "$RESULT_DIR/tsa_core.json" "$RESULT_DIR/tsa_core.log"'''
assert t.count(oldblock)==1;t=t.replace(oldblock,newblock,1)
Path('runpod/run_treatment_90_seeded_screen.sh').write_text(t);Path('runpod/run_treatment_90_seeded_screen.sh').chmod(0o755)
PY_PATCH

{
  diff -u config/experiment.env.original config/experiment.env || true
  diff -u runpod/run_treatment_x.sh.original runpod/run_treatment_90_seeded_screen.sh || true
  diff -u /dev/null tools/validate_direct9_manifest.py || true
} > /workspace/experiment_patch.diff

[[ "$(grep '^TSA_ALPHA=' config/experiment.env)" == 'TSA_ALPHA=0.587' ]]
[[ "$(grep '^SNAPSHOT_SPACING_FRAC=' config/experiment.env)" == 'SNAPSHOT_SPACING_FRAC=0.011108926464805349' ]]
bash -n runpod/run_treatment_90_seeded_screen.sh
python3 - <<'PY'
import json
from pathlib import Path
endpoint=10172;spacing=113;k=8;steps=[endpoint-spacing*i for i in range(k-1,-1,-1)]
m={'complete':True,'train_num_iterations':10172,'schedule_num_iterations':10172,'snapshot_k':8,'snapshot_spacing_steps':113,'required_steps':steps,'missing_steps':[],'captured':{str(x):{} for x in steps},'endpoints':{'9':{'ratio':9.0,'endpoint_step':endpoint,'snapshot_steps':steps}},'user_config':{'target_param_data_ratio':9.0,'record_schedule_param_data_ratio':9.0,'record_terminal_lr_clamp_frac':0.15}}
Path('/tmp/direct9_manifest_preflight.json').write_text(json.dumps(m))
PY
python3 tools/validate_direct9_manifest.py --manifest /tmp/direct9_manifest_preflight.json --ratio 9.0 --clamp 0.15 --k 8 --spacing 113 | tee /workspace/manifest_validator_preflight.txt

status "PHASE 3/6: one-time NanoChat/data setup and seed instrumentation"
if ! bash runpod/setup_once.sh; then status "FATAL: setup_once.sh failed"; exit 23; fi
REPO="$ROOT/nanochat-record"
cp "$REPO/scripts/base_train.py" /workspace/base_train.before_seed_patch.py
python3 - "$REPO/scripts/base_train.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
arg='parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")\n'
assert s.count(arg)==1
s=s.replace(arg,arg+'parser.add_argument("--experiment-seed", type=int, default=42, help="model-initialization RNG seed for replicated runs")\n',1)
anchor='ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)\n'
assert s.count(anchor)==1
seed=anchor+'''# D22 missing-row replication: override NanoChat's default initialization seed (42).\ntorch.manual_seed(args.experiment_seed)\nif device_type == "cuda":\n    torch.cuda.manual_seed(args.experiment_seed)\nprint0(f"Experiment initialization seed: {args.experiment_seed}")\n'''
s=s.replace(anchor,seed,1)
p.write_text(s)
PY
diff -u /workspace/base_train.before_seed_patch.py "$REPO/scripts/base_train.py" > /workspace/seed_patch.diff || true
grep -q -- '--experiment-seed' "$REPO/scripts/base_train.py"
grep -q 'Experiment initialization seed:' "$REPO/scripts/base_train.py"
source "$REPO/.venv/bin/activate"
python -m py_compile "$REPO/scripts/base_train.py"
python -m pip freeze > /workspace/pip-freeze.txt 2>/dev/null || true

printf 'seed,configuration,time_min,val_bpb,core\n' > "$FRESH"

run_cell() {
  local seed="$1"
  local config="floor15_direct9_tsa0587" script="$SUITE/runpod/run_treatment_90_seeded_screen.sh" recipe='blend:0.587'
  local tag log result
  tag="n3_${config}_s${seed}"
  log="/workspace/n3_logs/${tag}.stdout.log"
  result="$ROOT/results/$tag"
  status "RUN START seed=$seed config=$config"
  touch /workspace/any_training_started.flag
  rm -rf "$result"
  set +e
  EXPERIMENT_SEED="$seed" RUN_TAG="$tag" KEEP_CHECKPOINTS=1 bash "$script" 9.0 > "$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then status "FATAL: seed=$seed config=$config script exited $rc"; tail -100 "$log" || true; exit 30; fi
  [[ -d "$result" ]] || { status "FATAL: missing result dir $result"; exit 31; }
  local j; j="$(find_core_json "$result" "$recipe" || true)"
  [[ -n "$j" ]] || { status "FATAL: result JSON recipe=$recipe not found in $result"; exit 31; }
  local step val core tm
  step="$(json_field "$j" step)"; val="$(json_field "$j" val_bpb)"; core="$(json_field "$j" core_metric)"; tm="$(json_field "$j" training_minutes)"
  [[ "$step" == "10172" ]] || { status "FATAL: wrong endpoint step $step"; exit 32; }
  grep -q "Experiment initialization seed: $seed" "$log" || { status "FATAL: seed audit missing in $log"; exit 32; }
  grep -q "Record schedule ratio: 9.0000" "$log" || { status "FATAL: direct-9.0 schedule audit missing"; exit 32; }
  python3 "$SUITE/tools/validate_direct9_manifest.py" --manifest "$result/snapshot_manifest.json" --ratio 9.0 --clamp 0.15 --k 8 --spacing 113 > "/workspace/n3_logs/${tag}.manifest_audit.txt"
  grep -q 'blend:0.587' "$result/candidate.csv" || { status "FATAL: candidate manifest lacks blend:0.587"; exit 32; }
  [[ -f "$result/alpha_screen/bpb_screen.csv" ]] || { status "FATAL: alpha screen missing"; exit 32; }
  python3 - "$j" "$COMPACT/${tag}.json" "$seed" "$config" <<'PY'
import json,sys
src,out,seed,config=sys.argv[1:]
d=json.load(open(src));d['seed']=int(seed);d['configuration']=config
json.dump(d,open(out,'w'),indent=2,sort_keys=True);open(out,'a').write('\n')
PY
  cp "$result/alpha_screen/bpb_screen.csv" "$COMPACT/${tag}_alpha_screen.csv"
  printf '%s,%s,%.9f,%.9f,%.9f\n' "$seed" "$config" "$tm" "$val" "$core" >> "$FRESH"
  copy_meta_and_hash_model "$tag" "$j"
  cleanup_checkpoint_dir_from_json "$j"
  rm -rf "$ROOT/data/base_checkpoints/${tag}_train" 2>/dev/null || true
  status "RUN DONE seed=$seed config=$config time_min=$tm val_bpb=$val core=$core"
  df -h /workspace > "/workspace/n3_logs/${tag}.disk_after.txt" || true
}

status "PHASE 4/6: three direct-9.0 + 15% floor + fixed TSA(.587) runs, with descriptive alpha screen"
for seed in "${SEEDS[@]}"; do run_cell "$seed"; done

status "PHASE 5/6: summarize primary missing row and the descriptive alpha grid"
cp "$FRESH" "$ALL"
python3 - "$FRESH" "$N3" "$LATEX" "$SUMMARY" "$COMPACT" "$ALPHA_ALL" "$ALPHA_SUM" "$ALPHA_REC" <<'PY'
import csv,math,statistics,sys
from pathlib import Path
fresh,sump,texp,txtp,compact,alpha_all,alpha_sum,alpha_rec=map(Path,sys.argv[1:])
rows=list(csv.DictReader(fresh.open()))
assert [int(r['seed']) for r in rows]==[42,43,44]
for r in rows:
    for k in ('time_min','val_bpb','core'):r[k]=float(r[k])
tcrit=4.302652729911275
summary={}
with sump.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['configuration','metric','n','mean','sd','se','ci95_halfwidth','ci95_low','ci95_high','core_qualifies'])
    for metric in ('time_min','val_bpb','core'):
        xs=[r[metric] for r in rows];mean=statistics.mean(xs);sd=statistics.stdev(xs);se=sd/math.sqrt(3);hw=tcrit*se
        q=sum(x>=0.256525 for x in xs) if metric=='core' else ''
        w.writerow(['floor15_direct9_tsa0587',metric,3,f'{mean:.9f}',f'{sd:.9f}',f'{se:.9f}',f'{hw:.9f}',f'{mean-hw:.9f}',f'{mean+hw:.9f}',q]);summary[metric]=(mean,hw,sd,xs,q)
screen=[]
for seed in (42,43,44):
    p=compact/f'n3_floor15_direct9_tsa0587_s{seed}_alpha_screen.csv'
    for r in csv.DictReader(p.open()):
        rec=dict(r);rec['seed']=seed;screen.append(rec)
with alpha_all.open('w',newline='') as f:
    fields=['seed']+[x for x in screen[0] if x!='seed'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(screen)
def alpha_of(recipe):return float(recipe.split(':',1)[1])
by={}
for r in screen:by.setdefault(alpha_of(r['recipe']),[]).append(float(r['screen_val_bpb']))
with alpha_sum.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['alpha','n','mean_screen_val_bpb','sd','se','ci95_halfwidth','delta_vs_alpha_0.587'])
    ref=statistics.mean(by[0.587])
    for a in sorted(by):
        xs=by[a];mean=statistics.mean(xs);sd=statistics.stdev(xs);se=sd/math.sqrt(3);hw=tcrit*se
        w.writerow([f'{a:.6f}',3,f'{mean:.9f}',f'{sd:.9f}',f'{se:.9f}',f'{hw:.9f}',f'{mean-ref:+.9f}'])
pooled=min(sorted(by),key=lambda a:(statistics.mean(by[a]),abs(a-0.587),a))
seed_best={}
for seed in (42,43,44):
    rr=[r for r in screen if int(r['seed'])==seed]
    best=min(rr,key=lambda r:(float(r['screen_val_bpb']),abs(alpha_of(r['recipe'])-0.587),alpha_of(r['recipe'])))
    seed_best[seed]=alpha_of(best['recipe'])
loo={}
for held in (42,43,44):
    others=[s for s in (42,43,44) if s!=held]
    loo[held]=min(sorted(by),key=lambda a:(statistics.mean([float(r['screen_val_bpb']) for r in screen if int(r['seed']) in others and alpha_of(r['recipe'])==a]),abs(a-0.587),a))
rec=[
'Alpha-screen diagnostic for direct-9.0 + 15% floor trajectories',
'===============================================================',
'PRIMARY ROW IS PREDECLARED TSA alpha=0.587. The screen does not alter it.',
'The screen uses 4,194,304 BPB tokens per candidate and is descriptive.',
'Because alpha is screened on these same three trained trajectories, the pooled minimizer is not an independent confirmation result.',
'',
f'pooled discrete mean-BPB minimizer: alpha={pooled:.6f}',
'per-seed screen minimizers: '+', '.join(f'seed{s}={seed_best[s]:.6f}' for s in (42,43,44)),
'leave-one-seed-out screen choices: '+', '.join(f'holdout{s}->{loo[s]:.6f}' for s in (42,43,44)),
'',
'If the pooled/LOO choices cluster near 0.587, retain 0.587. If they move materially and consistently, use a separate development trajectory to freeze a new alpha before a confirmatory rerun.'
]
alpha_rec.write_text('\n'.join(rec)+'\n')
tm,th,*_=summary['time_min'];vm,vh,*_=summary['val_bpb'];cm,ch,*_=summary['core']
lines=[r'\begin{table}[h]',r'    \centering',r'    \caption{\textbf{Direct-calibration depth-22 treatment control.} Three fresh single-node 8$\times$H100 RunPod repetitions. The schedule is calibrated directly to ratio $9.0$; the terminal floor is $15\%$; the returned estimator is frozen at TSA $\alpha=0.587$.}',r'    \label{tab:d22-direct9-treatment}',r'    \begin{tabular}{lccccc}',r'        \toprule',r'        Configuration & Calibration & Floor & Time (min) & Val. BPB $\downarrow$ & CORE $\uparrow$ \\',r'        \midrule',f'        15\% floor + TSA (direct 9.0) & $9.0$ & $15\%$ & ${tm:.2f} \pm {th:.2f}$ & ${vm:.6f} \pm {vh:.6f}$ & ${cm:.4f} \pm {ch:.4f}$ \\',r'        \bottomrule',r'    \end{tabular}',r'\end{table}']
texp.write_text('\n'.join(lines)+'\n')
out=['D22 missing-row RunPod N=3 summary','====================================','Configuration: direct ratio-9.0 schedule + 15% floor + TSA(alpha=0.587)','Checkpoint window held fixed at steps 9381,9494,9607,9720,9833,9946,10059,10172.','']
for m in ('time_min','val_bpb','core'):
    mean,hw,sd,xs,q=summary[m];extra=f' ; qualifies={q}/3' if m=='core' else ''
    out.append(f'{m}: mean={mean:.9f} +/- {hw:.9f} (95% t-CI; sd={sd:.9f}; values={xs}){extra}')
out += ['',alpha_rec.read_text().strip()]
txtp.write_text('\n'.join(out)+'\n');print(txtp.read_text())
PY

status "PHASE 6/6: final validation and evidence packaging"
[[ "$(($(wc -l < "$FRESH")-1))" -eq 3 ]]
[[ "$(($(wc -l < "$ALPHA_ALL")-1))" -eq 39 ]]
df -h /workspace > /workspace/disk_final.txt
cat "$SUMMARY"
cat "$ALPHA_REC"
exit 0
REMOTE
chmod +x "$REMOTE_WORKER_LOCAL"

# Validate the generated worker locally before any remote training can begin. bash -n
# catches syntax errors; the grep guard catches the exact nounset pattern that caused v1
# to fail at Phase 4 before the first run.
bash -n "$REMOTE_WORKER_LOCAL"
if grep -Eq 'local[[:space:]]+tag=.*\$\{?tag' "$REMOTE_WORKER_LOCAL"; then
  log "ERROR: generated worker contains unsafe same-statement tag self-reference."
  runpodctl pod delete "$POD_ID" >/dev/null 2>&1 || runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
  cleanup_key
  exit 7
fi
log "Generated remote worker passed syntax and nounset-path preflight."

log "Uploading remote worker..."
cat "$REMOTE_WORKER_LOCAL" | "${SSH_BASE[@]}" 'cat > /workspace/n3_remote_worker.sh && chmod +x /workspace/n3_remote_worker.sh'

log "Starting remote worker detached from this Mac connection..."
"${SSH_BASE[@]}" 'nohup bash /workspace/n3_remote_worker.sh >/dev/null 2>&1 < /dev/null & echo $! > /workspace/n3_remote_worker.pid'
WORKER_STARTED=1

log "Remote worker started. Polling once per minute."
LAST_STATUS=""
STOPPED_EARLY=0
while true; do
  # First try the done marker over the current SSH route.
  if "${SSH_BASE[@]}" 'test -f /workspace/n3_worker.done' >/dev/null 2>&1; then
    log "Remote worker wrote its completion marker; checking exit status."
    break
  fi

  POD_JSON="$(runpodctl pod list --all --name "$RUN_NAME" -o json 2>/dev/null || true)"
  POD_STATUS="$(printf '%s' "$POD_JSON" | python3 -c '
import json,sys
try:d=json.load(sys.stdin)
except:sys.exit()
if isinstance(d,dict):
  d=d.get("pods",[d]) if isinstance(d.get("pods",[d]),list) else [d]
for x in d or []:
  if isinstance(x,dict):
    print(x.get("desiredStatus") or x.get("status") or x.get("runtimeStatus") or "");break
' 2>/dev/null || true)"

  if echo "$POD_STATUS" | grep -qiE 'EXITED|STOPPED'; then
    log "WARNING: pod stopped before the worker wrote its done marker (server-side 10h failsafe or infrastructure stop)."
    log "Restarting briefly only to retrieve partial evidence; the three-run worker will NOT be silently restarted."
    runpodctl pod start "$POD_ID" >/dev/null 2>&1 || true
    sleep 45
    SSH_INFO="$(runpodctl ssh info "$POD_ID" 2>&1 || true)"
    PARSED="$(printf '%s\n' "$SSH_INFO" | python3 -c '
import json,sys,shlex
try:d=json.load(sys.stdin)
except:sys.exit(1)
ip=d.get("ip") or d.get("host") or "";port=d.get("port") or "";cmd=d.get("ssh_command") or d.get("sshCommand") or ""
if (not ip or not port) and cmd:
  try:t=shlex.split(cmd)
  except:t=cmd.split()
  for x in t:
    if "@" in x and not x.startswith("-"):ip=x.split("@",1)[1];break
  for i,x in enumerate(t[:-1]):
    if x=="-p":port=t[i+1]
if not ip or not port:sys.exit(1)
print(ip);print(port)
' 2>/dev/null || true)"
    NEW_IP="$(printf '%s\n' "$PARSED" | sed -n '1p')"; NEW_PORT="$(printf '%s\n' "$PARSED" | sed -n '2p')"
    if [[ -n "$NEW_IP" && -n "$NEW_PORT" ]]; then
      SSH_TARGET="root@$NEW_IP"; SSH_PORT="$NEW_PORT"
      SSH_BASE=(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$SSH_PORT" "$SSH_TARGET")
      "${SSH_BASE[@]}" 'cd /workspace && tar -czf /workspace/n3_emergency_evidence.tar.gz --ignore-failed-read n3_status.txt n3_remote_worker.log FINAL_SUMMARY.txt FRESH_RESULTS.csv N3_ALL_RUNS.csv N3_SUMMARY.csv N3_TABLE.tex ALPHA_SCREEN_ALL.csv ALPHA_SCREEN_SUMMARY.csv ALPHA_RECOMMENDATION.txt n3_logs n3_compact model_evidence 2>/dev/null || true' || true
      "${SSH_BASE[@]}" 'cat /workspace/n3_emergency_evidence.tar.gz 2>/dev/null || true' > "$STATE_DIR/emergency_evidence.tar.gz" || true
    fi
    runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
    log "POD PRESERVED STOPPED: $POD_ID"
    notify "D22 missing-row pod stopped before completion; partial evidence preserved"
    STOPPED_EARLY=1
    break
  fi

  # Refresh SSH routing in case RunPod changed the proxy endpoint.
  SSH_INFO="$(runpodctl ssh info "$POD_ID" 2>&1 || true)"
  PARSED="$(printf '%s\n' "$SSH_INFO" | python3 -c '
import json,sys,shlex
try:d=json.load(sys.stdin)
except:sys.exit(1)
ip=d.get("ip") or d.get("host") or "";port=d.get("port") or "";cmd=d.get("ssh_command") or d.get("sshCommand") or ""
if (not ip or not port) and cmd:
  try:t=shlex.split(cmd)
  except:t=cmd.split()
  for x in t:
    if "@" in x and not x.startswith("-"):ip=x.split("@",1)[1];break
  for i,x in enumerate(t[:-1]):
    if x=="-p":port=t[i+1]
if not ip or not port:sys.exit(1)
print(ip);print(port)
' 2>/dev/null || true)"
  NEW_IP="$(printf '%s\n' "$PARSED" | sed -n '1p')"; NEW_PORT="$(printf '%s\n' "$PARSED" | sed -n '2p')"
  if [[ -n "$NEW_IP" && -n "$NEW_PORT" ]]; then
    SSH_TARGET="root@$NEW_IP"; SSH_PORT="$NEW_PORT"
    SSH_BASE=(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$SSH_PORT" "$SSH_TARGET")
  fi

  CUR_STATUS="$("${SSH_BASE[@]}" 'tail -1 /workspace/n3_status.txt 2>/dev/null || true' 2>/dev/null || true)"
  if [[ -n "$CUR_STATUS" && "$CUR_STATUS" != "$LAST_STATUS" ]]; then
    log "REMOTE: $CUR_STATUS"; LAST_STATUS="$CUR_STATUS"
  fi
  sleep 60
done

if [[ "$STOPPED_EARLY" == "1" ]]; then
  log "Stopped before completion. State directory: $STATE_DIR"
  exit 8
fi

WORKER_DONE_LOCAL="$STATE_DIR/n3_worker.done"
"${SSH_BASE[@]}" 'cat /workspace/n3_worker.done 2>/dev/null || true' > "$WORKER_DONE_LOCAL" || true
PRESERVE_POD="$(awk -F= '$1=="preserve_pod"{print $2}' "$WORKER_DONE_LOCAL" 2>/dev/null | tail -1)"
WORKER_EXIT_CODE="$(awk -F= '$1=="exit_code"{print $2}' "$WORKER_DONE_LOCAL" 2>/dev/null | tail -1)"

for spec in \
  "FINAL_SUMMARY.txt:$LOCAL_SUMMARY" \
  "FRESH_RESULTS.csv:$LOCAL_FRESH" \
  "N3_ALL_RUNS.csv:$LOCAL_ALL" \
  "N3_SUMMARY.csv:$LOCAL_N3" \
  "N3_TABLE.tex:$LOCAL_LATEX"; do
  remote_name="${spec%%:*}"; local_name="${spec#*:}"
  "${SSH_BASE[@]}" "cat /workspace/$remote_name 2>/dev/null || true" > "$local_name" || true
done
for remote_name in ALPHA_SCREEN_ALL.csv ALPHA_SCREEN_SUMMARY.csv ALPHA_RECOMMENDATION.txt; do
  "${SSH_BASE[@]}" "cat /workspace/$remote_name 2>/dev/null || true" > "$STATE_DIR/$remote_name" || true
done
cat "$LOCAL_SUMMARY" 2>/dev/null || true

log "Downloading compact evidence archive..."
if "${SSH_BASE[@]}" 'cat /workspace/d22_n3_evidence.tar.gz' > "$LOCAL_EVIDENCE"; then
  EVIDENCE_SHA="$(shasum -a 256 "$LOCAL_EVIDENCE" | awk '{print $1}')"
  printf '%s  %s\n' "$EVIDENCE_SHA" "$(basename "$LOCAL_EVIDENCE")" > "$STATE_DIR/evidence.sha256"
  log "Evidence downloaded: $LOCAL_EVIDENCE"
else
  log "WARNING: evidence archive download failed; creating emergency archive."
  "${SSH_BASE[@]}" 'cd /workspace && tar -czf /workspace/n3_emergency_evidence.tar.gz --ignore-failed-read n3_status.txt n3_remote_worker.log FINAL_SUMMARY.txt FRESH_RESULTS.csv N3_ALL_RUNS.csv N3_SUMMARY.csv N3_TABLE.tex ALPHA_SCREEN_ALL.csv ALPHA_SCREEN_SUMMARY.csv ALPHA_RECOMMENDATION.txt n3_logs n3_compact model_evidence 2>/dev/null || true' || true
  "${SSH_BASE[@]}" 'cat /workspace/n3_emergency_evidence.tar.gz 2>/dev/null || true' > "$STATE_DIR/emergency_evidence.tar.gz" || true
fi

if [[ -z "$WORKER_EXIT_CODE" ]]; then
  log "ERROR: worker completion marker did not contain an exit_code; stopping pod for inspection."
  runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
  notify "D22 missing-row ended with an invalid completion marker; pod stopped"
  exit 9
fi

if [[ "$WORKER_EXIT_CODE" != "0" ]]; then
  if [[ "$PRESERVE_POD" == "1" ]]; then
    log "FAILED: worker exited $WORKER_EXIT_CODE after training began. Stopping pod and preserving disk: $POD_ID"
    runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true
    notify "D22 missing-row failed after training began; pod stopped and preserved"
  else
    log "FAILED: worker exited $WORKER_EXIT_CODE before useful training. Deleting pod to stop billing."
    if runpodctl pod delete "$POD_ID" >/dev/null 2>&1; then CLEANED=1; else runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true; fi
    cleanup_key
    notify "D22 missing-row failed before useful training; evidence downloaded"
  fi
  log "Failure state directory: $STATE_DIR"
  log "Evidence: $LOCAL_EVIDENCE"
  exit 9
fi

log "Deleting pod $POD_ID to stop billing after successful completion."
if runpodctl pod delete "$POD_ID" >/dev/null 2>&1; then CLEANED=1; else runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || true; fi
cleanup_key
notify "D22 missing-row complete; results downloaded and pod deleted"

log "DONE SUCCESS. State directory: $STATE_DIR"
log "Summary: $LOCAL_SUMMARY"
log "Fresh rows: $LOCAL_FRESH"
log "N=3 summary: $LOCAL_N3"
log "LaTeX table: $LOCAL_LATEX"
log "Alpha screen summary: $STATE_DIR/ALPHA_SCREEN_SUMMARY.csv"
log "Alpha recommendation: $STATE_DIR/ALPHA_RECOMMENDATION.txt"
log "Evidence: $LOCAL_EVIDENCE"
