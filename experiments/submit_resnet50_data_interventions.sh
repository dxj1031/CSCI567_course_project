#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${VARIANT_DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, VARIANT_DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

BBOX_BG_DATA_ROOT="$VARIANT_DATA_ROOT/dataset_bbox_bg"
HISTMATCH_DATA_ROOT="$VARIANT_DATA_ROOT/dataset_histmatch"

echo "Submitting ResNet50 bbox background-intervention runs..."
DATA_ROOT="$BBOX_BG_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50_bbox_bg.yaml
DATA_ROOT="$BBOX_BG_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50_bbox_bg.yaml
DATA_ROOT="$BBOX_BG_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50_bbox_bg.yaml

echo "Submitting ResNet50 histogram-matching runs..."
DATA_ROOT="$HISTMATCH_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50_histmatch.yaml
DATA_ROOT="$HISTMATCH_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50_histmatch.yaml
DATA_ROOT="$HISTMATCH_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50_histmatch.yaml
