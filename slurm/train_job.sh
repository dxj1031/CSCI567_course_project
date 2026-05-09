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
if [[ -n "${SEED:-}" ]]; then
  CMD+=(--seed "$SEED")
fi
if [[ -n "${LOSS:-}" ]]; then
  CMD+=(--loss "$LOSS")
fi
if [[ -n "${CLASS_WEIGHT_MODE:-}" ]]; then
  CMD+=(--class-weight-mode "$CLASS_WEIGHT_MODE")
fi
if [[ -n "${TRAIN_SAMPLER:-}" ]]; then
  CMD+=(--train-sampler "$TRAIN_SAMPLER")
fi
if [[ -n "${FOCAL_GAMMA:-}" ]]; then
  CMD+=(--focal-gamma "$FOCAL_GAMMA")
fi
if [[ -n "${IMAGE_ABLATION:-}" ]]; then
  CMD+=(--image-ablation "$IMAGE_ABLATION")
fi
if [[ -n "${IMAGE_ABLATION_BBOX_PADDING_FRACTION:-}" ]]; then
  CMD+=(--image-ablation-bbox-padding-fraction "$IMAGE_ABLATION_BBOX_PADDING_FRACTION")
fi
if [[ -n "${IMAGE_ABLATION_MASK_FEATHER:-}" ]]; then
  CMD+=(--image-ablation-mask-feather "$IMAGE_ABLATION_MASK_FEATHER")
fi
if [[ -n "${IMAGE_ABLATION_FILL_COLOR:-}" ]]; then
  CMD+=(--image-ablation-fill-color "$IMAGE_ABLATION_FILL_COLOR")
fi

echo "Running on $(hostname)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
