#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${SOURCE_DATA_ROOT:-}" || -z "${VARIANT_DATA_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, SOURCE_DATA_ROOT, and VARIANT_DATA_ROOT before running."
  exit 1
fi

export PYTHONNOUSERSITE=1

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/data_processing/background_intervention.py" \
  --source-root "$SOURCE_DATA_ROOT" \
  --output-root "$VARIANT_DATA_ROOT" \
  --variant-name dataset_bg_blur

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/data_processing/brightness_alignment.py" \
  --source-root "$SOURCE_DATA_ROOT" \
  --output-root "$VARIANT_DATA_ROOT" \
  --variant-name dataset_brightness_aligned
