#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
VARIANTS=(original photometric_randomization background_perturbation combined)

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      config="configs/${scenario}_${backbone}.yaml"
      if [[ "$variant" == "original" ]]; then
        train_intervention="none"
      else
        train_intervention="$variant"
      fi
      echo "Submitting $config with train_intervention=$train_intervention"
      DATA_ROOT="$DATA_ROOT" TRAIN_INTERVENTION="$train_intervention" bash "$PROJECT_ROOT/scripts/submit_train.sh" "$config"
    done
  done
done
