In this work we are exploring ways to detect defective parts in automotive production lines. Using the AutoVI dataset, we implement a PatchCore based detection loop and we attempt to apply trustworthy AI techniques to improve it.

## Install prerequisites ##
pip install -r requirements.txt
For fetching the dataset: curl, unzip, and sha256sum 

## Fetch data ##
To retrieve the AutoVI datase run:
./src/data/download_datasets.sh --data-root data

## Running the centralized learning script on local machine ##
 ./src/centralized/patchcore_training.py --dataset-root data/raw --categories engine_wiring

## Running the stage1 federated learning script on local machine ##
Start the server first:

./src/federated-stage1/server.py \
    --num-clients 3 \
    --num-rounds 3 \
    --server-address "0.0.0.0:8080" \
    --output-dir artifacts/federated/server

Then start 3 clients, make sure each has different ids and different output dirs:

CATEGORIES=(engine_wiring pipe_clip)
./src/federated-stage1/client.py \
    --client-id 1 \
    --server "0.0.0.0:8080" \
    --categories "${CATEGORIES[@]}" \
    --dataset-root data/raw \
    --output-dir artifacts/federated/client1 \
    --coreset-method random \
    --coreset-ratio 0.1 \
    --batch-size 4 \
    --image-size 512 \
    --coreset-max-samples 5000 \
    --num-workers 4 \
    --coreset-chunk-size 16384 \
    --distance-chunk-size 8192 \
    --log-level INFO \
    --device cuda \

## Running the final federated learning script on local machine ##
Start the server first:

./src/federated/server_sequential.py /
  --host 0.0.0.0 --port 8081 /
  --expected-client-ids 1 2 3 /
  --server-max-samples 5000 /
  --server-coreset-chunk-size 16384 /
  --output-dir federated_sequentially/artifacts/federated_sequential/federated_improved/server /
  --log-level INFO
  
Then start the clients, make sure each has different ids and output dirs:

./src/federated/client_sequential.py /
  --mode upload --client-id 1 --categories engine_wiring pipe_clip /
  --server-host 10.8.0.5 --server-port 8081 /
  --dataset-root data/raw /
  --output-dir federated_sequentially/artifacts/federated_sequential/federated_improved/clients /
  --image-size 512 --batch-size 4 --num-workers 4 /
  --coreset-method random --coreset-ratio 0.1 --coreset-max-samples 5000 /
  --coreset-chunk-size 16384 --distance-chunk-size 8192 /
  --train-partition-id 0 --train-num-partitions 2 /
  --device cuda --log-level INFO

After all clients have finished, run the eval mode for all clients:

./src/federated/client_sequential.py /
  --mode eval --client-id 1 --categories engine_wiring pipe_clip /
  --server-host 10.8.0.5 --server-port 8081 /
  --dataset-root data/raw /
  --output-dir federated_sequentially/artifacts/federated_sequential/federated_improved/clients /
  --image-size 512 --batch-size 4 --num-workers 4 /
  --coreset-method random --coreset-ratio 0.1 --coreset-max-samples 5000 /
  --coreset-chunk-size 16384 --distance-chunk-size 8192 /
  --train-partition-id 0 --train-num-partitions 2 /
  --device cuda --log-level INFO

### Optional: Client-side Differential Privacy (DP)
Enable DP on upload to clip memory bank vectors and add calibrated noise:
```
./src/federated/client_sequential.py \
  --mode upload --client-id 1 --categories engine_wiring pipe_clip \
  --server-host 10.8.0.5 --server-port 8081 \
  --dataset-root data/raw \
  --output-dir federated_sequentially/artifacts/federated_sequential/federated_improved/clients \
  --dp --dp-epsilon 2.0 --dp-delta 1e-5 --dp-clip-norm 1.0
```
DP budget is written to `client_<id>_dp_budget.json` under the client output directory.

### Optional: Interpretability Outputs (Grad-CAM + SHAP)
Generate anomaly overlays and Grad-CAM-style saliency maps during evaluation:
```
./src/federated/client_sequential.py \
  --mode eval --client-id 1 --categories engine_wiring pipe_clip \
  --server-host 10.8.0.5 --server-port 8081 \
  --dataset-root data/raw \
  --output-dir federated_sequentially/artifacts/federated_sequential/federated_improved/clients \
  --interpretability --saliency-max-images 10
```
For SHAP-based post-hoc explanations over patch embeddings, install `shap` and set:
```
--shap-max-images 5 --shap-background 20 --shap-max-patches 64
```
Outputs are saved under `artifacts/.../interpretability/<category>/`.


## Adding Fairness-Aware Coreset Selection

This implementation adds fairness-aware coreset selection to Federated PatchCore to prevent bias towards categories/subgroups with more training data.

### Without Fairness:
```
Category A: 1000 images → 90% of coreset
Category B: 100 images  → 10% of coreset

Results:
  - Category A: AUROC = 0.95 ✓
  - Category B: AUROC = 0.67 ✗ (under-represented!)
```

## Available Modes

### 1. `none` - No Fairness (Baseline)
```bash
--fairness-coreset-mode none
```
- Applies k-Center on **all patches together**
- **Risk:** Minority groups are under-represented
- **Use case:** Baseline for comparison

---

### 2. `proportional` - Proportional Fairness
```bash
--fairness-coreset-mode proportional
```
- **Allocates patches proportionally** to subgroup size
- **Example:** 
  - Subgroup with 80% of patches → gets ~80% of coreset
  - Subgroup with 20% of patches → gets ~20% of coreset
- **Advantage:** Maintains natural distribution while ensuring minority representation
- **Use case:** Balanced accuracy and fairness

---

### 3. `equal` - Equal Fairness
```bash
--fairness-coreset-mode equal
```
- **Allocates equal number of patches** to each subgroup
- **Example:**
  - 2 subgroups, 5000 total patches → 2500 per subgroup
- **Advantage:** Maximum representation for minority groups
- **Disadvantage:** May under-represent majority groups
- **Use case:** When minority group performance is critical

---
