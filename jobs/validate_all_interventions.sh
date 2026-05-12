#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT before running."
  exit 1
fi

bash "$PROJECT_ROOT/jobs/validate_legacy_alignment.sh"
bash "$PROJECT_ROOT/jobs/validate_interventions.sh"
bash "$PROJECT_ROOT/jobs/validate_visibility.sh"
bash "$PROJECT_ROOT/jobs/validate_followup.sh"
