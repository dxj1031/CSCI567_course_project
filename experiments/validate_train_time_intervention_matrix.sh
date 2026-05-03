#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
VARIANTS=(original photometric_randomization background_perturbation combined)

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONNOUSERSITE=1

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      if [[ "$variant" == "original" ]]; then
        config="configs/${scenario}_${backbone}.yaml"
        train_intervention="none"
      else
        config="configs/${scenario}_${backbone}.yaml"
        train_intervention="$variant"
      fi
      echo "Validating $config with train_intervention=$train_intervention"
      "$ENV_PREFIX/bin/python" scripts/train_baseline.py \
        --config "$config" \
        --train-intervention "$train_intervention" \
        --validate-only
    done
  done
done
