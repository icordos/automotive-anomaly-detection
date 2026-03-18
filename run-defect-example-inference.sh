#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/raw}"
CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS:-$REPO_ROOT/artifacts/clients}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/defect-example-inference}"
DEVICE="${DEVICE:-cuda}"

CATEGORIES=(
  engine_wiring
  pipe_clip
  pipe_staple
  tank_screw
  underbody_pipes
  underbody_screw
)

python3 "$REPO_ROOT/src/federated/infer_defect_examples.py" \
  --dataset-root "$DATASET_ROOT" \
  --categories "${CATEGORIES[@]}" \
  --checkpoint-roots "$CHECKPOINT_ROOTS" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --shap-max-images 0 \
  --log-level INFO \
  "$@"
