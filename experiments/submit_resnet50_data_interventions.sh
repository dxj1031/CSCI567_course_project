#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${ORIGINAL_DATA_ROOT:-}" || -z "${VARIANT_DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, ORIGINAL_DATA_ROOT, VARIANT_DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SAM_DATA_ROOT="$VARIANT_DATA_ROOT/dataset_sam_bg"
HISTMATCH_DATA_ROOT="$VARIANT_DATA_ROOT/dataset_histmatch"

echo "Submitting ResNet50 baseline on original dataset..."
DATA_ROOT="$ORIGINAL_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50.yaml
DATA_ROOT="$ORIGINAL_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50.yaml
DATA_ROOT="$ORIGINAL_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50.yaml

echo "Submitting ResNet50 SAM background-intervention runs..."
DATA_ROOT="$SAM_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50_sam_bg.yaml
DATA_ROOT="$SAM_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50_sam_bg.yaml
DATA_ROOT="$SAM_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50_sam_bg.yaml

echo "Submitting ResNet50 histogram-matching runs..."
DATA_ROOT="$HISTMATCH_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/cross_location_resnet50_histmatch.yaml
DATA_ROOT="$HISTMATCH_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/day_to_night_resnet50_histmatch.yaml
DATA_ROOT="$HISTMATCH_DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" configs/night_to_day_resnet50_histmatch.yaml
