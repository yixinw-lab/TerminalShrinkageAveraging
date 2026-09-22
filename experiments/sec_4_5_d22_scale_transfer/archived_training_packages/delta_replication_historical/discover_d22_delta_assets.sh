#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-$HOME/ExtrapProj}"
BASE_COMMIT=e09bc164162f35da0b5b8315be791e9e974a4c3a
printf '===== REPOSITORY =====\n%s\n\n' "$REPO_ROOT"
printf '===== EXISTING EXACT-COMMIT NANOCHAT CHECKOUTS =====\n'
while IFS= read -r d; do
  [[ -d "$d/.git" ]] || continue
  h=$(git -C "$d" rev-parse HEAD 2>/dev/null || true)
  [[ "$h" == "$BASE_COMMIT" ]] && echo "$d"
done < <(find "$REPO_ROOT" /work/nvme/bhji/"$USER" -maxdepth 5 -type d -name nanochat 2>/dev/null | sort -u)
printf '\n===== OLD D22 / RUNPOD ARTIFACT HINTS =====\n'
find "$REPO_ROOT" /work/nvme/bhji/"$USER" -maxdepth 7 \( -iname '*d22*' -o -iname '*record*attempt*' -o -iname '*runpod*' \) -print 2>/dev/null | head -200 || true
printf '\n===== STORAGE =====\n'
df -h /work/nvme/bhji/"$USER" 2>/dev/null || true
