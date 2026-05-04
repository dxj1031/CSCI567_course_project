#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "Set PROJECT_ROOT before running."
  exit 1
fi

bash "$PROJECT_ROOT/experiments/submit_visibility_test_only.sh"
bash "$PROJECT_ROOT/experiments/submit_visibility_train_test_consistent.sh"
bash "$PROJECT_ROOT/experiments/submit_visibility_night_only.sh"
