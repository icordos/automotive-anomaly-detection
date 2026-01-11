# How to Run Federated PatchCore

This guide explains how to set up and run the Federated PatchCore system.

## Prerequisites
-   **Python 3.11** is highly recommended. (Newer versions like 3.14 may have compatibility issues with `protobuf` and `flwr`).
-   **Virtual Environment**: It is best practice to use a virtual environment.

## 1. Setup

Create and activate a virtual environment (using Python 3.11):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Running the System

You will need **three separate terminal windows** to simulate the federation (1 Server + 2 Clients).

### Terminal 1: Start the Server
The server coordinates the training.
```bash
# Make sure your venv is activated
source .venv/bin/activate

# Start the server (default port 8080)
python federated_server.py --dataset-root data/raw --categories engine_wiring
```
*Note: If port 8080 is in use, the script might fail. You can edit `federated_server.py` and `federated_client.py` to change the port.*

### Terminal 2: Start Client 1
```bash
source .venv/bin/activate

# Start the first client
python federated_client.py --dataset-root data/raw --categories engine_wiring
```

### Terminal 3: Start Client 2
```bash
source .venv/bin/activate

# Start the second client
python federated_client.py --dataset-root data/raw --categories engine_wiring
```

## 3. What to Expect
1.  **Server**: Will wait until 2 clients connect.
2.  **Training**: Once connected, the server instructs clients to train.
3.  **Clients**: Will extract features from their local data (this may take a minute).
4.  **Aggregation**: Clients send memory banks to the server. The server aggregates them and applies coreset subsampling.
5.  **Evaluation**: The global model is sent back to clients for evaluation.
6.  **Repeat**: This process repeats for 3 rounds (configured in `federated_server.py`).

## Troubleshooting
-   **Port Conflicts**: If you see "address already in use" errors, ensure no other process is using port 8080. You can kill lingering python processes with `pkill -f python`.
-   **Protobuf Errors**: If you see errors related to `google.protobuf`, ensure you are using Python 3.11 and have a clean environment.
