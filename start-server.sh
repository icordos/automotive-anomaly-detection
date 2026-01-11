./src/federated/server_sequential.py \
  --host 0.0.0.0 --port 8081 \
  --expected-client-ids 1 2 3 \
  --server-max-samples 5000 \
  --server-coreset-chunk-size 16384 \
  --output-dir artifacts\server \
  --log-level INFO