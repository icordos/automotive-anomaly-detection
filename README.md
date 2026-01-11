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

Then start the clients, make sure each has different ids and output dirs:


./src/federated_sequentially/client_sequential.py /
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
./src/federated_sequentially/client_sequential.py /
  --mode eval --client-id 1 --categories engine_wiring pipe_clip /
  --server-host 10.8.0.5 --server-port 8081 /
  --dataset-root data/raw /
  --output-dir federated_sequentially/artifacts/federated_sequential/federated_improved/clients /
  --image-size 512 --batch-size 4 --num-workers 4 /
  --coreset-method random --coreset-ratio 0.1 --coreset-max-samples 5000 /
  --coreset-chunk-size 16384 --distance-chunk-size 8192 /
  --train-partition-id 0 --train-num-partitions 2 /
  --device cuda --log-level INFO
