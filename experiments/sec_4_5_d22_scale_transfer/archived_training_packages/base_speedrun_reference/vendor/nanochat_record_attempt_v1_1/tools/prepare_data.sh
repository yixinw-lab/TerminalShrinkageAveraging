#!/usr/bin/env bash
set -euo pipefail
REPO=${1:-"$HOME/nanochat-record"}
SHARDS=${SHARDS:-170}
VOCAB_SIZE=${VOCAB_SIZE:-49152}
: "${NANOCHAT_BASE_DIR:?export NANOCHAT_BASE_DIR to a persistent shared path first}"
cd "$REPO"
source .venv/bin/activate
python -m nanochat.dataset -n "$SHARDS"
python -m scripts.tok_train --vocab-size="$VOCAB_SIZE"
python -m scripts.tok_eval
printf '\nPrepared data/tokenizer under %s\n' "$NANOCHAT_BASE_DIR"
