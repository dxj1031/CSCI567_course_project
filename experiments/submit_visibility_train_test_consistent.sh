#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
MODES=(original gamma clahe gamma_clahe)
NIGHT_ONLY_FLAG_SOURCE="${NIGHT_ONLY_FLAG_SOURCE:-day_night}"

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for mode in "${MODES[@]}"; do
      config="configs/${scenario}_${backbone}.yaml"
      echo "Submitting train+test consistent visibility run: $config mode=$mode"
      JOB_NAME="cct20_vis_train" \
        TRAIN_INTERVENTION="none" \
        VISIBILITY_SCOPE="train_test_consistent" \
        VISIBILITY_MODE="$mode" \
        NIGHT_ONLY_FLAG_SOURCE="$NIGHT_ONLY_FLAG_SOURCE" \
        DATA_ROOT="$DATA_ROOT" \
        bash "$PROJECT_ROOT/scripts/submit_train.sh" "$config"
    done
  done
done
