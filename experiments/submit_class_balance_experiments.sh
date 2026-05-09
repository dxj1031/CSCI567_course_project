#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ACCOUNT:-}" || -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set ACCOUNT, PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

IFS=' ' read -r -a SEEDS_ARRAY <<< "${SEEDS:-42}"
SCENARIOS=(cross_location day_to_night night_to_day)
METHODS=(baseline_weighted no_class_weights class_balanced_sampler focal_balanced)

winner_backbone() {
  case "$1" in
    cross_location) echo "resnet101" ;;
    day_to_night) echo "resnet101" ;;
    night_to_day) echo "resnet50" ;;
    *) echo "Unknown scenario: $1" >&2; exit 1 ;;
  esac
}

apply_method_env() {
  local method="$1"
  LOSS="cross_entropy"
  CLASS_WEIGHT_MODE="balanced"
  TRAIN_SAMPLER="none"
  FOCAL_GAMMA=""

  case "$method" in
    baseline_weighted)
      ;;
    no_class_weights)
      CLASS_WEIGHT_MODE="none"
      ;;
    class_balanced_sampler)
      CLASS_WEIGHT_MODE="none"
      TRAIN_SAMPLER="class_balanced"
      ;;
    focal_balanced)
      LOSS="focal"
      CLASS_WEIGHT_MODE="balanced"
      FOCAL_GAMMA="${FOCAL_GAMMA_DEFAULT:-2.0}"
      ;;
    *)
      echo "Unknown method: $method" >&2
      exit 1
      ;;
  esac
  export LOSS CLASS_WEIGHT_MODE TRAIN_SAMPLER FOCAL_GAMMA
}

for scenario in "${SCENARIOS[@]}"; do
  backbone="$(winner_backbone "$scenario")"
  config="configs/${scenario}_${backbone}.yaml"
  for method in "${METHODS[@]}"; do
    apply_method_env "$method"
    for seed in "${SEEDS_ARRAY[@]}"; do
      echo "Submitting class-balance follow-up: config=$config method=$method seed=$seed"
      SEED="$seed" TRAIN_INTERVENTION=none bash "$PROJECT_ROOT/scripts/submit_train.sh" "$config"
    done
  done
done
