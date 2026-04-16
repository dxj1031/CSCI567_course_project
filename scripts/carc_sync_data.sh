#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DRIVE_REMOTE:-}" || -z "${DRIVE_PATH:-}" || -z "${DEST_ROOT:-}" ]]; then
  echo "Set DRIVE_REMOTE, DRIVE_PATH, and DEST_ROOT before running."
  echo "Example:"
  echo "  export DRIVE_REMOTE=mydrive"
  echo "  export DRIVE_PATH=shared/CS567/CCT20"
  echo "  export DEST_ROOT=/project2/<PI>_<project_id>/datasets/CCT20"
  exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is not available in PATH. Configure it on CARC first."
  exit 1
fi

mkdir -p "$DEST_ROOT"
rclone copy "${DRIVE_REMOTE}:${DRIVE_PATH}" "$DEST_ROOT" --progress --create-empty-src-dirs

echo "Dataset copy complete: ${DRIVE_REMOTE}:${DRIVE_PATH} -> $DEST_ROOT"

