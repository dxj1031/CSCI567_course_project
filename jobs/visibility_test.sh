#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
MODES=(original gamma clahe gamma_clahe)
BASELINE_RESULTS_ROOT="${BASELINE_RESULTS_ROOT:-$OUTPUT_ROOT}"
NIGHT_ONLY_FLAG_SOURCE="${NIGHT_ONLY_FLAG_SOURCE:-day_night}"

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for mode in "${MODES[@]}"; do
      config="configs/${scenario}_${backbone}.yaml"
      echo "Submitting test-only visibility eval: $config mode=$mode checkpoint_root=$BASELINE_RESULTS_ROOT"
      JOB_NAME="cct20_vis_eval" \
        TRAIN_INTERVENTION="none" \
        VISIBILITY_SCOPE="test_only" \
        VISIBILITY_MODE="$mode" \
        NIGHT_ONLY_FLAG_SOURCE="$NIGHT_ONLY_FLAG_SOURCE" \
        CHECKPOINT_RESULTS_ROOT="$BASELINE_RESULTS_ROOT" \
        DATA_ROOT="$DATA_ROOT" \
        bash "$PROJECT_ROOT/scripts/submit_train.sh" "$config"
    done
  done
done
