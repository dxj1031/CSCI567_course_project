#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
VARIANTS=(original bbox_blur brightness_aligned)

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONNOUSERSITE=1

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      if [[ "$variant" == "original" ]]; then
        config="configs/${scenario}_${backbone}.yaml"
      else
        config="configs/${scenario}_${backbone}_${variant}.yaml"
      fi
      echo "Validating $config"
      "$ENV_PREFIX/bin/python" scripts/train_baseline.py --config "$config" --validate-only
    done
  done
done
