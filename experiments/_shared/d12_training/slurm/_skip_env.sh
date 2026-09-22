#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -f "${REPO}/env.sh" ]]; then
  # Existing project variables take precedence when available.
  # shellcheck disable=SC1090
  source "${REPO}/env.sh"
fi

if [[ -z "${NANOCHAT_ROOT:-}" ]]; then
  for candidate in \
    "${HOME}/nanochat" \
    "/scratch/bhgh/${USER}/nanochat" \
    "/scratch/${USER}/nanochat"; do
    if [[ -f "${candidate}/nanochat/gpt.py" ]]; then
      NANOCHAT_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${NANOCHAT_ROOT:-}" || ! -f "${NANOCHAT_ROOT}/nanochat/gpt.py" ]]; then
  echo "Could not locate NanoChat. Export NANOCHAT_ROOT=/path/to/nanochat and resubmit." >&2
  exit 2
fi
if [[ ! -f "${NANOCHAT_ROOT}/.venv/bin/activate" ]]; then
  echo "NanoChat virtualenv missing at ${NANOCHAT_ROOT}/.venv" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${NANOCHAT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO}:${NANOCHAT_ROOT}:${PYTHONPATH:-}"
export REPO NANOCHAT_ROOT
mkdir -p "${REPO}/logs" "${REPO}/outputs"
cd "${REPO}"
