#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
VARIANTS=(blur bright)

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      config="configs/legacy/${scenario}_${backbone}_${variant}.yaml"
      echo "Submitting legacy alignment intervention: $config"
      bash "$PROJECT_ROOT/scripts/submit_train.sh" "$config"
    done
  done
done
