#!/usr/bin/env bash
set -u
DOWNLOADS="${DOWNLOADS:-$HOME/Downloads}"; LOG="$DOWNLOADS/d22_runpod_alpha70_supervisor.log"; LATEST="$DOWNLOADS/d22_runpod_alpha70_latest"
echo '===== SUPERVISOR ====='; pgrep -fl 'runpod_d22_alpha70_supervisor_v[0-9]+\.sh|caffeinate.*runpod_d22_alpha70' || echo 'not running'
echo; echo '===== LATEST LOG ====='; tail -60 "$LOG" 2>/dev/null || echo "no log yet: $LOG"
echo
if [[ -L "$LATEST" || -d "$LATEST" ]]; then echo '===== LATEST STATE ====='; echo "$(readlink "$LATEST" 2>/dev/null || echo "$LATEST")"; [[ -f "$LATEST/FINAL_SUMMARY.txt" ]] && { echo; cat "$LATEST/FINAL_SUMMARY.txt"; }; [[ -f "$LATEST/FINAL_THREE_ROW_TABLE.tex" ]] && { echo; echo "===== FINAL THREE-ROW TABLE ====="; cat "$LATEST/FINAL_THREE_ROW_TABLE.tex"; }; fi
echo; echo '===== RUNPOD ====='; runpodctl pod list --all 2>/dev/null || runpodctl pod list 2>/dev/null || true
