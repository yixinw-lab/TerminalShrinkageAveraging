#!/usr/bin/env bash
set -u
DOWNLOADS="${DOWNLOADS:-$HOME/Downloads}"
PIDFILE="$DOWNLOADS/d22_runpod_n3_supervisor.pid"
LATEST="$DOWNLOADS/d22_runpod_n3_latest"

if [[ -f "$PIDFILE" ]]; then
  SUP="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$SUP" ]]; then
    for c in $(pgrep -P "$SUP" 2>/dev/null); do kill -TERM "$c" 2>/dev/null || true; done
    kill -TERM "$SUP" 2>/dev/null || true
  fi
fi
# Catch an orphaned supervisor child, but not this stop script.
pkill -TERM -f '[r]unpod_d22_n3_supervisor_v[0-9]+\.sh' 2>/dev/null || true
sleep 2

echo '===== SUPERVISOR CHECK ====='
pgrep -fl 'runpod_d22_n3_supervisor_v[0-9]+\.sh|caffeinate.*runpod_d22_n3' || echo 'supervisor stopped'

POD=""
if [[ -f "$LATEST/pod_id.txt" ]]; then POD="$(cat "$LATEST/pod_id.txt" 2>/dev/null || true)"; fi
if [[ -n "$POD" ]]; then
  echo
  echo "Stopping RunPod pod $POD (preserving disk; NOT deleting)..."
  runpodctl pod stop "$POD" 2>/dev/null || true
fi

echo
echo '===== RUNPOD CHECK ====='
runpodctl pod list --all 2>/dev/null || runpodctl pod list 2>/dev/null || true
