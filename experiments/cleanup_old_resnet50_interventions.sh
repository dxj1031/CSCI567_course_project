#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VARIANT_DATA_ROOT:-}" ]]; then
  echo "Set VARIANT_DATA_ROOT before running."
  exit 1
fi

ROOT_PATH="$(realpath "$VARIANT_DATA_ROOT")"
TARGETS=(
  "$ROOT_PATH/dataset_bg_blur"
  "$ROOT_PATH/dataset_brightness_aligned"
)

for TARGET in "${TARGETS[@]}"; do
  case "$TARGET" in
    "$ROOT_PATH/dataset_bg_blur"|"$ROOT_PATH/dataset_brightness_aligned")
      ;;
    *)
      echo "Refusing to delete unexpected path: $TARGET"
      exit 1
      ;;
  esac

  if [[ -d "$TARGET" ]]; then
    rm -rf "$TARGET"
    echo "Deleted $TARGET"
  else
    echo "Skipping missing $TARGET"
  fi
done
