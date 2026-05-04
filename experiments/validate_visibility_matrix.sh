#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${DATA_ROOT:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, DATA_ROOT, and OUTPUT_ROOT before running."
  exit 1
fi

SCENARIOS=(cross_location day_to_night night_to_day)
BACKBONES=(resnet18 resnet34 resnet50 resnet101)
SCOPES=(test_only train_test_consistent night_only)
MODES=(original gamma clahe gamma_clahe)
NIGHT_ONLY_FLAG_SOURCE="${NIGHT_ONLY_FLAG_SOURCE:-day_night}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONNOUSERSITE=1

for scenario in "${SCENARIOS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for scope in "${SCOPES[@]}"; do
      for mode in "${MODES[@]}"; do
        config="configs/${scenario}_${backbone}.yaml"
        echo "Validating visibility matrix: $config scope=$scope mode=$mode"
        "$ENV_PREFIX/bin/python" scripts/train_baseline.py \
          --config "$config" \
          --train-intervention none \
          --visibility-scope "$scope" \
          --visibility-mode "$mode" \
          --night-only-flag-source "$NIGHT_ONLY_FLAG_SOURCE" \
          --validate-only
      done
    done
  done
done
