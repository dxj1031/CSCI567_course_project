#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT before running."
  exit 1
fi

bash "$PROJECT_ROOT/jobs/legacy_alignment.sh"
bash "$PROJECT_ROOT/jobs/train_interventions.sh"
bash "$PROJECT_ROOT/jobs/visibility_all.sh"
bash "$PROJECT_ROOT/jobs/object_diagnostics.sh"
