#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
VARIANTS=("" bbox_blur brightness_aligned)

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      if [[ -z "$variant" ]]; then
        config="configs/${scenario}_${backbone}.yaml"
      else
        config="configs/${scenario}_${backbone}_${variant}.yaml"
      fi
      echo "Submitting $config"
      DATA_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/scripts/submit_train.sh" "$config"
    done
  done
done
