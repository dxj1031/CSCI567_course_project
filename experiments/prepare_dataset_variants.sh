#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${SOURCE_DATA_ROOT:-}" || -z "${VARIANT_DATA_ROOT:-}" || -z "${SAM_CHECKPOINT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, SOURCE_DATA_ROOT, VARIANT_DATA_ROOT, and SAM_CHECKPOINT before running."
  exit 1
fi

export PYTHONNOUSERSITE=1

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/data_processing/background_intervention.py" \
  --source-root "$SOURCE_DATA_ROOT" \
  --output-root "$VARIANT_DATA_ROOT" \
  --variant-name dataset_sam_bg \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --model-type "${SAM_MODEL_TYPE:-vit_h}" \
  --device "${SAM_DEVICE:-auto}"

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/data_processing/brightness_alignment.py" \
  --source-root "$SOURCE_DATA_ROOT" \
  --output-root "$VARIANT_DATA_ROOT" \
  --variant-name dataset_histmatch \
  --target-mode "${HISTMATCH_TARGET_MODE:-combined_train}"
