#!/usr/bin/env python
"""Run the sequential federated PatchCore pipeline (1 round, server k-center aggregation).

This script:
1) starts server_sequential.py
2) runs clients sequentially in upload mode (client1 -> client2 -> client3)
3) runs clients sequentially in eval mode (client1 -> client2 -> client3)
4) shuts down the server

IMPROVED:
- Writes logs/artifacts under federated_sequential/{logs|artifacts}/federated_sequential/federated_improved/...
- Splits at least one category across 2 clients via partition args (requires client/dataset support)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _python_exe() -> str:
    return sys.executable


def _popen(cmd: List[str], out_path: Path, err_path: Path) -> subprocess.Popen:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8", buffering=1)
    err_f = open(err_path, "w", encoding="utf-8", buffering=1)
    return subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)


def _terminate(p: subprocess.Popen, name: str) -> None:
    try:
        logging.info("Terminating %s (pid=%s)", name, p.pid)
        p.terminate()
        p.wait(timeout=10)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def _rpc_shutdown(server_host: str, server_port: int) -> None:
    import socket, pickle, struct

    def send_msg(sock, obj):
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        sock.sendall(struct.pack("!Q", len(payload)) + payload)

    def recv_msg(sock):
        header = sock.recv(8)
        if not header:
            return {}
        (length,) = struct.unpack("!Q", header)
        buf = b""
        while len(buf) < length:
            chunk = sock.recv(length - len(buf))
            if not chunk:
                break
            buf += chunk
        return pickle.loads(buf) if buf else {}

    with socket.create_connection((server_host, server_port), timeout=10.0) as sock:
        send_msg(sock, {"action": "shutdown"})
        _ = recv_msg(sock)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sequential federated PatchCore pipeline (improved)")
    p.add_argument("--server-host", type=str, default="127.0.0.1")
    p.add_argument("--server-bind-host", type=str, default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=8081)

    p.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("federated_sequentially/artifacts/federated_sequential"))
    p.add_argument("--logs-dir", type=Path, default=Path("federated_sequentially/logs/federated_sequential"))

    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--coreset-method", type=str, default="random", choices=["random", "kcenter"])
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--coreset-max-samples", type=int, default=5000)
    p.add_argument("--coreset-chunk-size", type=int, default=16384)
    p.add_argument("--distance-chunk-size", type=int, default=8192)

    p.add_argument("--server-max-samples", type=int, default=5000)
    p.add_argument("--server-coreset-chunk-size", type=int, default=16384)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--dp", action="store_true", help="Enable client-side DP before upload")
    p.add_argument("--dp-epsilon", type=float, default=1.0)
    p.add_argument("--dp-delta", type=float, default=1e-5)
    p.add_argument("--dp-clip-norm", type=float, default=1.0)
    p.add_argument("--dp-seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(message)s")

    artifacts_dir = args.artifacts_dir / "federated_improved"
    logs_dir = args.logs_dir / "federated_improved"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    server_out = logs_dir / "server_out.txt"
    server_err = logs_dir / "server_err.txt"

    server_cmd = [
        _python_exe(),
        "-u",
        "federated_sequentially/server_sequential.py",
        "--host", args.server_bind_host,
        "--port", str(args.server_port),
        "--expected-client-ids", "1", "2", "3",
        "--server-max-samples", str(args.server_max_samples),
        "--server-coreset-chunk-size", str(args.server_coreset_chunk_size),
        "--output-dir", str(artifacts_dir / "server"),
        "--log-level", args.log_level,
    ]

    logging.info("Starting server")
    logging.info("CMD: %s", " ".join(server_cmd))
    server_p = _popen(server_cmd, server_out, server_err)

    time.sleep(2)

    # Split a category between 2 clients:
    # - engine_wiring split into 2 partitions: client1 gets partition 0/2, client2 gets partition 1/2
    # NOTE: This REQUIRES client_sequential.py + dataset loader to support these args:
    #   --train-partition-id, --train-num-partitions
    # (You asked earlier for splitting by hash/index; hook it there.)
    clients = [
        {"client_id": 1, "categories": ["engine_wiring", "pipe_clip"], "partition_id": 0, "num_partitions": 2},
        {"client_id": 2, "categories": ["engine_wiring", "tank_screw"], "partition_id": 1, "num_partitions": 2},
        {"client_id": 3, "categories": ["underbody_pipes", "underbody_screw"], "partition_id": None, "num_partitions": None},
    ]

    common_client_args = [
        "--server-host", args.server_host,
        "--server-port", str(args.server_port),
        "--dataset-root", str(args.dataset_root),
        "--output-dir", str(artifacts_dir / "clients"),
        "--image-size", str(args.image_size),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--coreset-method", args.coreset_method,
        "--coreset-ratio", str(args.coreset_ratio),
        "--coreset-max-samples", str(args.coreset_max_samples),
        "--coreset-chunk-size", str(args.coreset_chunk_size),
        "--distance-chunk-size", str(args.distance_chunk_size),
        "--device", args.device,
        "--log-level", args.log_level,
    ]
    if args.dp:
        common_client_args.extend(
            [
                "--dp",
                "--dp-epsilon", str(args.dp_epsilon),
                "--dp-delta", str(args.dp_delta),
                "--dp-clip-norm", str(args.dp_clip_norm),
            ]
        )
        if args.dp_seed is not None:
            common_client_args.extend(["--dp-seed", str(args.dp_seed)])

    # Upload phase (sequential)
    for c in clients:
        name = f"client{c['client_id']}_upload"
        outp = logs_dir / f"{name}_out.txt"
        errp = logs_dir / f"{name}_err.txt"
        cmd = [
            _python_exe(),
            "-u",
            "federated_sequentially/client_sequential.py",
            "--mode", "upload",
            "--client-id", str(c["client_id"]),
            "--categories", *c["categories"],
            *common_client_args,
        ]
        if c["partition_id"] is not None and c["num_partitions"] is not None:
            cmd += [
                "--train-partition-id", str(c["partition_id"]),
                "--train-num-partitions", str(c["num_partitions"]),
            ]
        logging.info("Starting %s", name)
        logging.info("CMD: %s", " ".join(cmd))
        p = _popen(cmd, outp, errp)
        rc = p.wait()
        if rc != 0:
            logging.error("%s exited with code %s. Check %s", name, rc, errp)
            _terminate(server_p, "server")
            return

    # Eval phase (sequential)
    for c in clients:
        name = f"client{c['client_id']}_eval"
        outp = logs_dir / f"{name}_out.txt"
        errp = logs_dir / f"{name}_err.txt"
        cmd = [
            _python_exe(),
            "-u",
            "federated_sequentially/client_sequential.py",
            "--mode", "eval",
            "--client-id", str(c["client_id"]),
            "--categories", *c["categories"],
            *common_client_args,
        ]
        if c["partition_id"] is not None and c["num_partitions"] is not None:
            cmd += [
                "--train-partition-id", str(c["partition_id"]),
                "--train-num-partitions", str(c["num_partitions"]),
            ]
        logging.info("Starting %s", name)
        logging.info("CMD: %s", " ".join(cmd))
        p = _popen(cmd, outp, errp)
        rc = p.wait()
        if rc != 0:
            logging.error("%s exited with code %s. Check %s", name, rc, errp)
            _terminate(server_p, "server")
            return

    # Shutdown server cleanly
    try:
        _rpc_shutdown(args.server_host, args.server_port)
    except Exception as e:
        logging.warning("Failed to shutdown server via RPC: %s", e)

    rc = server_p.wait(timeout=10)
    logging.info("Server exited with code %s", rc)
    logging.info("Done. Metrics are in %s", artifacts_dir / "clients")


if __name__ == "__main__":
    main()
