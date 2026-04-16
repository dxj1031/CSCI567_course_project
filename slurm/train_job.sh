#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${CONFIG_PATH:-}" ]]; then
  echo "Missing PROJECT_ROOT, ENV_PREFIX, or CONFIG_PATH."
  exit 1
fi

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

CMD=(python scripts/train_baseline.py --config "$CONFIG_PATH")

if [[ "${SMOKE:-0}" == "1" ]]; then
  CMD+=(--smoke)
fi

echo "Running on $(hostname)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
