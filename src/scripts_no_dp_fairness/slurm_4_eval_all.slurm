#!/bin/bash
#SBATCH --job-name=fed_eval_all
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err

set -e

source /export/home/mecanica/stud/m/matei.neaga/miniconda3/bin/activate ro_auth_detect

SERVER_HOST="172.24.12.1"
SERVER_PORT=8081

DATASET_ROOT="../data/raw"

# Output dir WITHOUT DP (must match upload scripts)
OUTPUT_DIR="artifacts/federated_sequential/federated_improved/clients"

CORESET_METHOD="kcenter"
CORESET_RATIO=0.1
CORESET_MAX_SAMPLES=5000
CORESET_CHUNK_SIZE=16384

mkdir -p logs "$OUTPUT_DIR"

echo "========================================"
echo "Federated Evaluation - All Clients (NO DP)"
echo "========================================"
echo "Server: $SERVER_HOST:$SERVER_PORT"
echo "Output dir: $OUTPUT_DIR"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "========================================"

# ============================================
# Client 1: engine_wiring, pipe_clip
# ============================================
echo ""
echo "--- Evaluating Client 1 (engine_wiring, pipe_clip) ---"
python client_sequential.py \
    --mode eval \
    --client-id 1 \
    --categories engine_wiring pipe_clip \
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
    --train-partition-id 0 \
    --train-num-partitions 3 \
    --device cuda \
    --log-level INFO

echo "✓ Client 1 completed"

# ============================================
# Client 2: pipe_staple, tank_screw
# ============================================
echo ""
echo "--- Evaluating Client 2 (pipe_staple, tank_screw) ---"
python client_sequential.py \
    --mode eval \
    --client-id 2 \
    --categories pipe_staple tank_screw \
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
    --train-partition-id 1 \
    --train-num-partitions 3 \
    --device cuda \
    --log-level INFO

echo "✓ Client 2 completed"

# ============================================
# Client 3: underbody_pipes, underbody_screw
# ============================================
echo ""
echo "--- Evaluating Client 3 (underbody_pipes, underbody_screw) ---"
python client_sequential.py \
    --mode eval \
    --client-id 3 \
    --categories underbody_pipes underbody_screw \
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
    --train-partition-id 2 \
    --train-num-partitions 3 \
    --device cuda \
    --log-level INFO

echo "✓ Client 3 completed"

# ============================================
# Summary
# ============================================
echo ""
echo "========================================"
echo "All evaluations completed at $(date)"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/client_1/client_1_metrics.json"
echo "  - $OUTPUT_DIR/client_2/client_2_metrics.json"
echo "  - $OUTPUT_DIR/client_3/client_3_metrics.json"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/client_*/client_*_metrics.json | python -m json.tool"
echo ""
echo "To compare with DP version (if exists):"
echo "  diff <(cat artifacts/federated_sequential/federated_improved/clients/client_1/client_1_metrics.json) \\"
echo "       <(cat artifacts/federated_sequential/federated_improved_WITH_DP/clients/client_1/client_1_metrics.json)"
echo "========================================"