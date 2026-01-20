#!/bin/bash

# $1 captures the first parameter you type after the script name
# run first the script with upload then after all clients are ran with eval
MODE=$1

# Optional: Check if the user actually provided a mode
if [ -z "$MODE" ]; then
  echo "Error: You must provide a mode. Usage: ./start-client-1.sh <mode>"
  exit 1
fi

python ./src/federated/client_sequential.py \
    --mode "$MODE" \
    --client-id 3 \
    --categories underbody_pipes underbody_screw \
    --server-host 127.0.0.1 --server-port 8081 \
    --dataset-root data/raw \
    --output-dir artifacts/clients \
    --image-size 512 --batch-size 4 --num-workers 4 \
    --coreset-method kcenter --coreset-ratio 0.1 --coreset-max-samples 5000 \
    --coreset-chunk-size 16384 --distance-chunk-size 8192 \
    --train-partition-id 0 --train-num-partitions 2 \
    --interpretability --saliency-max-images 10 \
    --shap-max-images 5 --shap-background 20 --shap-max-patches 64 \
    --corruption-prob 0.2 --corruption-strength 0.3 --gaussian-noise-std 0.02 \
    --robust-norm-max 10.0 --robust-cosine-min 0.0 --anomaly-score-clip 10.0 \
    --device cuda --log-level INFO
