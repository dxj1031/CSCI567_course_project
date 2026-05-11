#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT before running."
  exit 1
fi

bash "$PROJECT_ROOT/jobs/visibility_test.sh"
bash "$PROJECT_ROOT/jobs/visibility_train_test.sh"
bash "$PROJECT_ROOT/jobs/visibility_night.sh"
