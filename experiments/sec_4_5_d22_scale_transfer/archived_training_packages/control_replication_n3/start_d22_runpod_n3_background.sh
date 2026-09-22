#!/usr/bin/env bash
set -euo pipefail
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set in this shell. Export it first.}"
command -v runpodctl >/dev/null || { echo 'ERROR: runpodctl not found'; exit 2; }
command -v caffeinate >/dev/null || { echo 'ERROR: caffeinate not found (this launcher is for macOS)'; exit 2; }
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUP="$BUNDLE_DIR/runpod_d22_n3_supervisor_v2.sh"
DOWNLOADS="${DOWNLOADS:-$HOME/Downloads}"
LOG="$DOWNLOADS/d22_runpod_n3_supervisor.log"
PIDFILE="$DOWNLOADS/d22_runpod_n3_supervisor.pid"

OLD="$(pgrep -fl 'runpod_supervisor_d22_paired_v1\.py|runpod_d22_virtual_horizon_supervisor_v[0-9]+\.sh|wait_for_8xh100\.sh|runpod_d22_n3_supervisor_v[0-9]+\.sh' 2>/dev/null || true)"
if [[ -n "$OLD" ]]; then
  echo 'ERROR: a RunPod D22 waiter/supervisor is already running:'
  echo "$OLD"
  echo 'Stop it before launching this one to avoid acquiring two 8xH100 pods.'
  exit 3
fi

mkdir -p "$DOWNLOADS/d22_runpod_n3_results"
nohup caffeinate -i bash "$SUP" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PIDFILE"
disown "$PID" 2>/dev/null || true

echo "Started D22 N=3 supervisor under caffeinate PID $PID"
echo "Log:     $LOG"
echo "Results: $DOWNLOADS/d22_runpod_n3_results"
echo "Latest:  $DOWNLOADS/d22_runpod_n3_latest"
echo "Watch:   tail -f '$LOG'"
echo "Status:  bash '$BUNDLE_DIR/status_d22_runpod_n3.sh'"
echo "Cancel:  bash '$BUNDLE_DIR/stop_d22_runpod_n3.sh'"
