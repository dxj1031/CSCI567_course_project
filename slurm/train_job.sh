#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${CONFIG_PATH:-}" ]]; then
  echo "Missing PROJECT_ROOT, ENV_PREFIX, or CONFIG_PATH."
  exit 1
fi

module purge
module load conda

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONNOUSERSITE=1

CMD=("$ENV_PREFIX/bin/python" scripts/train_baseline.py --config "$CONFIG_PATH")

if [[ "${SMOKE:-0}" == "1" ]]; then
  CMD+=(--smoke)
fi
if [[ -n "${TRAIN_INTERVENTION:-}" ]]; then
  CMD+=(--train-intervention "$TRAIN_INTERVENTION")
fi

echo "Running on $(hostname)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
