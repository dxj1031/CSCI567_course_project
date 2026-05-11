#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" ]]; then
  echo "Set ACCOUNT=<project_id> before running."
  exit 1
fi

PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

salloc \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT"

