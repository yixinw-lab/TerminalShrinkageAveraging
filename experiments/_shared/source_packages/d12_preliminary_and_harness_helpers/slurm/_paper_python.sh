#!/bin/bash
# Select the *NanoChat environment* Python, not a bare uv-managed interpreter.
# The suite needs Python >=3.9 plus numpy, torch, and matplotlib.
set -euo pipefail

_paper_python_ok() {
  local exe="$1"
  local resolved=""
  if [[ "$exe" == */* ]]; then
    [[ -x "$exe" ]] || return 1
    resolved="$exe"
  else
    resolved="$(command -v "$exe" 2>/dev/null || true)"
    [[ -n "$resolved" && -x "$resolved" ]] || return 1
  fi
  "$resolved" - <<'PY' >/dev/null 2>&1
import sys
if sys.version_info < (3, 9):
    raise SystemExit(1)
import numpy
import torch
import matplotlib
PY
}

paper_select_python() {
  local c resolved

  if [[ -n "${PAPER_PYTHON:-}" ]] && _paper_python_ok "$PAPER_PYTHON"; then
    if [[ "$PAPER_PYTHON" == */* ]]; then
      printf '%s\n' "$PAPER_PYTHON"
    else
      command -v "$PAPER_PYTHON"
    fi
    return 0
  fi

  # Prefer an activated/project virtualenv created by NanoChat / uv.
  for c in \
    "${VIRTUAL_ENV:-}/bin/python" \
    "$PWD/.venv/bin/python" \
    "${REPO:-$PWD}/.venv/bin/python" \
    "$HOME/ExtrapProj/.venv/bin/python" \
    python python3 python3.12 python3.11 python3.10 python3.9; do
    [[ "$c" == "/bin/python" ]] && continue
    if _paper_python_ok "$c"; then
      if [[ "$c" == */* ]]; then
        printf '%s\n' "$c"
      else
        command -v "$c"
      fi
      return 0
    fi
  done

  # Last resort: inspect uv-managed environments/venvs, but reject bare CPython
  # installs that do not contain the NanoChat dependencies.
  while IFS= read -r c; do
    if _paper_python_ok "$c"; then
      printf '%s\n' "$c"
      return 0
    fi
  done < <(find "$HOME" -maxdepth 5 -type f -path '*/bin/python*' -perm -u+x 2>/dev/null | head -200)

  cat >&2 <<'MSG'
No Python >=3.9 with numpy, torch, and matplotlib was found after loading the
NanoChat environment. Check the environment with:
  source slurm/_skip_env.sh
  command -v python || true
  command -v python3 || true
  python -c 'import numpy, torch, matplotlib; print("deps OK")' 2>/dev/null || true
Set PAPER_PYTHON=/path/to/the/NanoChat/python if needed.
MSG
  return 1
}

PAPER_PYTHON="$(paper_select_python)"
export PAPER_PYTHON
