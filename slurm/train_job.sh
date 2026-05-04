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
if [[ -n "${VISIBILITY_MODE:-}" ]]; then
  CMD+=(--visibility-mode "$VISIBILITY_MODE")
fi
if [[ -n "${VISIBILITY_SCOPE:-}" ]]; then
  CMD+=(--visibility-scope "$VISIBILITY_SCOPE")
fi
if [[ -n "${NIGHT_ONLY_FLAG_SOURCE:-}" ]]; then
  CMD+=(--night-only-flag-source "$NIGHT_ONLY_FLAG_SOURCE")
fi
if [[ -n "${VISIBILITY_GAMMA:-}" ]]; then
  CMD+=(--visibility-gamma "$VISIBILITY_GAMMA")
fi
if [[ -n "${VISIBILITY_CLAHE_CLIP_LIMIT:-}" ]]; then
  CMD+=(--visibility-clahe-clip-limit "$VISIBILITY_CLAHE_CLIP_LIMIT")
fi
if [[ -n "${VISIBILITY_CLAHE_TILE_GRID_SIZE:-}" ]]; then
  CMD+=(--visibility-clahe-tile-grid-size "$VISIBILITY_CLAHE_TILE_GRID_SIZE")
fi
if [[ -n "${EVAL_ONLY_CHECKPOINT:-}" ]]; then
  CMD+=(--eval-only-checkpoint "$EVAL_ONLY_CHECKPOINT")
fi
if [[ -n "${CHECKPOINT_RESULTS_ROOT:-}" ]]; then
  CMD+=(--checkpoint-results-root "$CHECKPOINT_RESULTS_ROOT")
fi

echo "Running on $(hostname)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
