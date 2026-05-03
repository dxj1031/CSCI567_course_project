#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

echo "Submitting ResNet50 train-time intervention runs on original validation/test data..."
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50_bbox_blur.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50_brightness_aligned.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50_bbox_blur.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50_brightness_aligned.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50_bbox_blur.yaml
DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50_brightness_aligned.yaml
