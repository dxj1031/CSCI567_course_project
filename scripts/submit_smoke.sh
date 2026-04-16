#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/submit_smoke.sh configs/cross_location_resnet18.yaml"
  exit 1
fi

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

CONFIG_PATH="$1"

sbatch \
  --account="$ACCOUNT" \
  --partition="${PARTITION:-gpu}" \
  --job-name="cct20_smoke" \
  --gres="gpu:${GPUS:-1}" \
  --cpus-per-task="${CPUS_PER_TASK:-4}" \
  --mem="${MEMORY:-32G}" \
  --time="${TIME_LIMIT:-00:30:00}" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",ENV_PREFIX="$ENV_PREFIX",DATA_ROOT="$DATA_ROOT",OUTPUT_ROOT="$OUTPUT_ROOT",CONFIG_PATH="$CONFIG_PATH",SMOKE=1 \
  "$PROJECT_ROOT/slurm/train_job.sh"

