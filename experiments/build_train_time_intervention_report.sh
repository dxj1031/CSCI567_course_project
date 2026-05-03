#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, and OUTPUT_ROOT before running."
  exit 1
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/train_time_interventions}"
mkdir -p "$ARTIFACT_ROOT"

export PYTHONNOUSERSITE=1

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/scripts/compare_interventions.py" \
  --results-root "$OUTPUT_ROOT" \
  --output-dir "$ARTIFACT_ROOT"

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/scripts/plot_intervention_results.py" \
  --comparison-dir "$ARTIFACT_ROOT" \
  --output-dir "$ARTIFACT_ROOT"
