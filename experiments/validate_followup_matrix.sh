#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONNOUSERSITE=1

SCENARIOS=(cross_location day_to_night night_to_day)
CLASS_BALANCE_METHODS=(baseline_weighted no_class_weights class_balanced_sampler focal_balanced)
IMAGE_ABLATIONS=(none object_crop foreground_only background_only)

winner_backbone() {
  case "$1" in
    cross_location) echo "resnet101" ;;
    day_to_night) echo "resnet101" ;;
    night_to_day) echo "resnet50" ;;
    *) echo "Unknown scenario: $1" >&2; exit 1 ;;
  esac
}

run_validate() {
  echo "Validating: $*"
  "$ENV_PREFIX/bin/python" scripts/train_baseline.py "$@" --validate-only
}

for scenario in "${SCENARIOS[@]}"; do
  backbone="$(winner_backbone "$scenario")"
  config="configs/${scenario}_${backbone}.yaml"

  run_validate --config "$config" --seed 42

  for method in "${CLASS_BALANCE_METHODS[@]}"; do
    case "$method" in
      baseline_weighted)
        run_validate --config "$config" --seed 42 --loss cross_entropy --class-weight-mode balanced --train-sampler none
        ;;
      no_class_weights)
        run_validate --config "$config" --seed 42 --loss cross_entropy --class-weight-mode none --train-sampler none
        ;;
      class_balanced_sampler)
        run_validate --config "$config" --seed 42 --loss cross_entropy --class-weight-mode none --train-sampler class_balanced
        ;;
      focal_balanced)
        run_validate --config "$config" --seed 42 --loss focal --class-weight-mode balanced --train-sampler none --focal-gamma 2.0
        ;;
    esac
  done

  for image_ablation in "${IMAGE_ABLATIONS[@]}"; do
    run_validate \
      --config "$config" \
      --seed 42 \
      --image-ablation "$image_ablation" \
      --image-ablation-bbox-padding-fraction "${IMAGE_ABLATION_BBOX_PADDING_FRACTION:-0.04}" \
      --image-ablation-mask-feather "${IMAGE_ABLATION_MASK_FEATHER:-2.0}" \
      --image-ablation-fill-color "${IMAGE_ABLATION_FILL_COLOR:-127,127,127}"
  done
done
