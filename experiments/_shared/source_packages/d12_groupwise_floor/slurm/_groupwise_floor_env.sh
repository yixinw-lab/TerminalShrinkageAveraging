#!/bin/bash
set -euo pipefail
unset PAPER_PYTHON
source "${REPO:?}/slurm/_skip_env.sh"
PAPER_PYTHON="${NANOCHAT_ROOT:?}/.venv/bin/python"
[[ -x "$PAPER_PYTHON" ]] || { echo "missing NanoChat Python: $PAPER_PYTHON" >&2; exit 2; }
"$PAPER_PYTHON" - <<'PY' >/dev/null
import numpy, torch, matplotlib, rustbpe, nanochat
PY
export PAPER_PYTHON
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
