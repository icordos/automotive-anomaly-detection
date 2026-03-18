#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/raw}"
CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS:-$REPO_ROOT/artifacts/clients}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/inference}"
DEVICE="${DEVICE:-cuda}"
PATCH_QUALITY_TOP_PERCENT="${PATCH_QUALITY_TOP_PERCENT:-1.0}"

CATEGORIES=(
  engine_wiring
  pipe_clip
  pipe_staple
  tank_screw
  underbody_pipes
  underbody_screw
)

python3 "$REPO_ROOT/src/federated/infer_test_folder.py" \
  --dataset-root "$DATASET_ROOT" \
  --categories "${CATEGORIES[@]}" \
  --checkpoint-roots "$CHECKPOINT_ROOTS" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --patch-quality-top-percent "$PATCH_QUALITY_TOP_PERCENT" \
  --log-level INFO \
  "$@"
