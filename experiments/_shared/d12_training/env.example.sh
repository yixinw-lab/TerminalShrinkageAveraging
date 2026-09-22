#!/bin/bash
# New portable template, NOT a reconstruction of the original machine environment.
# Copy to env.sh locally only after supplying these variables. env.sh is gitignored.
: "${ACCOUNT:?Export your authorized Slurm account before sourcing env.sh}"
: "${PARTITION:?Export the Slurm partition for your allocation}"
: "${NANOCHAT_ROOT:?Export the path to your matching NanoChat checkout}"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACCOUNT PARTITION NANOCHAT_ROOT
export PYTHONPATH="${REPO}:${NANOCHAT_ROOT}:${PYTHONPATH:-}"
# Set NANOCHAT_BASE_DIR separately to your tokenizer and dataset location.
# Preserve the original experiment environment/commit for exact reruns; not bundled.
