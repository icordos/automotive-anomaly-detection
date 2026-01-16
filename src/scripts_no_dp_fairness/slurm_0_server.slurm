#!/bin/bash
#SBATCH --job-name=fed_server
#SBATCH --partition=dgxh100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err

set -e

# ============================================
# Activate Conda Environment
# ============================================
source /export/home/mecanica/stud/m/matei.neaga/miniconda3/bin/activate ro_auth_detect

# ============================================
# Setup
# ============================================
mkdir -p logs
mkdir -p artifacts/federated_sequential/federated_improved/server

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')
SERVER_PORT=8081

echo "========================================"
echo "Federated PatchCore Server"
echo "========================================"
echo "Node: $(hostname)"
echo "IP: $SERVER_IP"
echo "Port: $SERVER_PORT"
echo "Expected Clients: 1, 2, 3"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Started at: $(date)"
echo "========================================"

# ============================================
# Start Server
# ============================================
python server_sequential.py \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --expected-client-ids 1 2 3 \
    --server-max-samples 5000 \
    --server-coreset-chunk-size 16384 \
    --output-dir artifacts/federated_sequential/federated_improved/server \
    --log-level INFO

echo ""
echo "========================================"
echo "Server shutdown at $(date)"
echo "========================================"
