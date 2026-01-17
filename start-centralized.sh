#!/bin/bash

python ./src/centralized/patchcore_training.py \
    --dataset-root data/raw \
    --output-dir artifacts/centralized \
    --categories engine_wiring pipe_clip \
    --image-size 512 --batch-size 4 --num-workers 4 \
    --coreset-method kcenter --coreset-ratio 0.1 --coreset-max-samples 5000 \
    --coreset-chunk-size 16384 --distance-chunk-size 8192 \
    --device mps --log-level INFO
