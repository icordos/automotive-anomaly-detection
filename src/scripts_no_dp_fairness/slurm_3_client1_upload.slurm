#!/bin/bash
#SBATCH --job-name=fed_c1_upload
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/client3_upload_%j.out
#SBATCH --error=logs/client3_upload_%j.err

set -e

source /export/home/mecanica/stud/m/matei.neaga/miniconda3/bin/activate ro_auth_detect

CLIENT_ID=3
SERVER_HOST="172.24.12.1"
SERVER_PORT=8081

CATEGORIES="underbody_pipes underbody_screw"
DATASET_ROOT="../data/raw"

# Output dir WITHOUT DP
OUTPUT_DIR="artifacts/federated_sequential/federated_improved/clients"

CORESET_METHOD="kcenter"
CORESET_RATIO=0.1
CORESET_MAX_SAMPLES=5000
CORESET_CHUNK_SIZE=16384

FAIRNESS_MODE="proportional"

TRAIN_PARTITION_ID=0
TRAIN_NUM_PARTITIONS=3

mkdir -p logs "$OUTPUT_DIR"

echo "========================================"
echo "Client $CLIENT_ID - Upload Phase (NO DP)"
echo "========================================"
echo "Server: $SERVER_HOST:$SERVER_PORT"
echo "Categories: $CATEGORIES"
echo "Fairness: $FAIRNESS_MODE"
echo "DP: DISABLED"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "========================================"

python client_sequential.py \
    --mode upload \
    --client-id "$CLIENT_ID" \
    --categories $CATEGORIES \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --image-size 512 \
    --batch-size 4 \
    --num-workers 4 \
    --coreset-method "$CORESET_METHOD" \
    --coreset-ratio "$CORESET_RATIO" \
    --coreset-max-samples "$CORESET_MAX_SAMPLES" \
    --coreset-chunk-size "$CORESET_CHUNK_SIZE" \
    --distance-chunk-size 8192 \
    --train-partition-id "$TRAIN_PARTITION_ID" \
    --train-num-partitions "$TRAIN_NUM_PARTITIONS" \
    --fairness-coreset-mode "$FAIRNESS_MODE" \
    --device cuda \
    --log-level INFO

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Client $CLIENT_ID upload completed successfully"
else
    echo "✗ Client $CLIENT_ID upload failed with code $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "========================================"

exit $EXIT_CODE
