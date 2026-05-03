#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || -z "${ENV_PREFIX:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT, ENV_PREFIX, and OUTPUT_ROOT before running."
  exit 1
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/train_time_diversification_$RUN_TIMESTAMP}"
if [[ -e "$ARTIFACT_ROOT" && "${ALLOW_EXISTING_ARTIFACT_ROOT:-0}" != "1" ]]; then
  echo "Refusing to reuse existing ARTIFACT_ROOT=$ARTIFACT_ROOT"
  echo "Set ARTIFACT_ROOT to a new directory, or set ALLOW_EXISTING_ARTIFACT_ROOT=1 if you intentionally want to update it."
  exit 1
fi
mkdir -p "$ARTIFACT_ROOT"

export PYTHONNOUSERSITE=1

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/scripts/compare_interventions.py" \
  --results-root "$OUTPUT_ROOT" \
  --output-dir "$ARTIFACT_ROOT"

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/scripts/plot_intervention_results.py" \
  --comparison-dir "$ARTIFACT_ROOT" \
  --output-dir "$ARTIFACT_ROOT"
