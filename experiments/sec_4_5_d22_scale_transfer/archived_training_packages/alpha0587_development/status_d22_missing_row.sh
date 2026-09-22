#!/usr/bin/env bash
set -u
DOWNLOADS="${DOWNLOADS:-$HOME/Downloads}"; LOG="$DOWNLOADS/d22_runpod_missing_row_supervisor.log"; LATEST="$DOWNLOADS/d22_runpod_missing_row_latest"
echo '===== SUPERVISOR ====='; pgrep -fl 'runpod_d22_missing_row_supervisor_v[0-9]+\.sh|caffeinate.*runpod_d22_missing_row' || echo 'not running'
echo; echo '===== LATEST LOG ====='; tail -60 "$LOG" 2>/dev/null || echo "no log yet: $LOG"
echo
if [[ -L "$LATEST" || -d "$LATEST" ]]; then echo '===== LATEST STATE ====='; echo "$(readlink "$LATEST" 2>/dev/null || echo "$LATEST")"; [[ -f "$LATEST/FINAL_SUMMARY.txt" ]] && { echo; cat "$LATEST/FINAL_SUMMARY.txt"; }; [[ -f "$LATEST/ALPHA_RECOMMENDATION.txt" ]] && { echo; cat "$LATEST/ALPHA_RECOMMENDATION.txt"; }; fi
echo; echo '===== RUNPOD ====='; runpodctl pod list --all 2>/dev/null || runpodctl pod list 2>/dev/null || true
