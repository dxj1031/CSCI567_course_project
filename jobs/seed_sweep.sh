#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

IFS=' ' read -r -a SEEDS_ARRAY <<< "${SEEDS:-42 43 44}"
SCENARIOS=(cross_location day_to_night night_to_day)

winner_backbone() {
  case "$1" in
    cross_location) echo "resnet101" ;;
    day_to_night) echo "resnet101" ;;
    night_to_day) echo "resnet50" ;;
    *) echo "Unknown scenario: $1" >&2; exit 1 ;;
  esac
}

for scenario in "${SCENARIOS[@]}"; do
  backbone="$(winner_backbone "$scenario")"
  config="configs/${scenario}_${backbone}.yaml"
  for seed in "${SEEDS_ARRAY[@]}"; do
    echo "Submitting seed sweep: config=$config seed=$seed"
    SEED="$seed" \
    TRAIN_INTERVENTION=none \
    LOSS=cross_entropy \
    CLASS_WEIGHT_MODE=balanced \
    TRAIN_SAMPLER=none \
      bash "$PROJECT_ROOT/scripts/submit_train.sh" "$config"
  done
done
