"""Sequential Federated PatchCore Server (no Flower).

- Keeps server online.
- Accepts clients sequentially.
- Clients upload local memory banks.
- After all expected clients uploaded, server aggregates per category using k-center greedy coreset.
- Clients later download the aggregated global memory banks.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import socket
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 8)
    (length,) = struct.unpack("!Q", header)
    payload = _recv_exact(sock, length)
    return pickle.loads(payload)


def send_msg(sock: socket.socket, obj: dict) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def kcenter_greedy_torch(features: np.ndarray, target: int, chunk_size: int = 16384, seed: int = 0) -> np.ndarray:
    if target <= 0:
        raise ValueError("target must be > 0")
    n = features.shape[0]
    if target >= n:
        return np.arange(n, dtype=np.int64)

    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)

    indices = torch.empty(target, dtype=torch.long, device=device)
    indices[0] = torch.randint(0, n, (1,), device=device)[0]
    min_dist = torch.full((n,), float("inf"), device=device)

    for i in range(1, target):
        last = x[indices[i - 1]].unsqueeze(0)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = x[start:end]
            d = torch.cdist(chunk, last).squeeze(1)
            min_dist[start:end] = torch.minimum(min_dist[start:end], d)
        indices[i] = torch.argmax(min_dist)

    return indices.cpu().numpy().astype(np.int64)


@dataclass
class ServerState:
    expected_client_ids: List[int]
    uploads: Dict[int, dict]
    global_ready: bool
    global_banks: Dict[str, np.ndarray]
    patch_shapes: Dict[str, Tuple[int, int]]


class SequentialFederatedServer:
    def __init__(
        self,
        host: str,
        port: int,
        expected_client_ids: List[int],
        server_max_samples: int,
        server_coreset_chunk_size: int,
        output_dir: Path,
    ) -> None:
        self.host = host
        self.port = port
        self.server_max_samples = server_max_samples
        self.server_coreset_chunk_size = server_coreset_chunk_size
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = ServerState(
            expected_client_ids=expected_client_ids,
            uploads={},
            global_ready=False,
            global_banks={},
            patch_shapes={},
        )

        self._shutdown_event = threading.Event()

    def _all_uploaded(self) -> bool:
        return all(cid in self.state.uploads for cid in self.state.expected_client_ids)

    def _aggregate_if_ready(self) -> None:
        if self.state.global_ready:
            return
        if not self._all_uploaded():
            return

        logging.info("[Server] All clients uploaded. Aggregating global memory banks...")

        category_to_banks: Dict[str, List[np.ndarray]] = {}
        patch_shapes: Dict[str, Tuple[int, int]] = {}

        for cid, payload in self.state.uploads.items():
            categories = payload["categories"]
            client_patch_shapes = payload.get("patch_shapes", {})
            banks = payload["memory_banks"]
            for cat in categories:
                mb = banks.get(cat)
                if mb is None or mb.size == 0:
                    continue
                category_to_banks.setdefault(cat, []).append(mb)
                if cat in client_patch_shapes:
                    patch_shapes[cat] = tuple(client_patch_shapes[cat])

        global_banks: Dict[str, np.ndarray] = {}
        for cat, mb_list in category_to_banks.items():
            concat = np.concatenate(mb_list, axis=0)
            logging.info("[Server] %s: received %d vectors (dim=%d) from %d clients",
                         cat, concat.shape[0], concat.shape[1], len(mb_list))
            if concat.shape[0] > self.server_max_samples:
                idx = kcenter_greedy_torch(
                    concat,
                    target=self.server_max_samples,
                    chunk_size=self.server_coreset_chunk_size,
                    seed=0,
                )
                concat = concat[idx]
                logging.info("[Server] %s: k-center coreset %d -> %d", cat, len(idx), concat.shape[0])
            global_banks[cat] = concat.astype(np.float32, copy=False)
            logging.info("[Server] %s: global bank shape %s", cat, global_banks[cat].shape)

        self.state.global_banks = global_banks
        self.state.patch_shapes = patch_shapes
        self.state.global_ready = True

        snapshot_path = self.output_dir / "global_memory_banks.pkl"
        with open(snapshot_path, "wb") as f:
            pickle.dump(
                {"global_banks": self.state.global_banks, "patch_shapes": self.state.patch_shapes},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logging.info("[Server] Saved global banks snapshot to %s", snapshot_path)

    def _handle_client(self, conn: socket.socket, addr: tuple) -> None:
        with conn:
            req = recv_msg(conn)
            action = req.get("action")

            if action == "upload":
                client_id = int(req["client_id"])
                logging.info("[Server] upload from client_id=%s addr=%s", client_id, addr)

                if client_id not in self.state.expected_client_ids:
                    send_msg(conn, {"ok": False, "error": f"unexpected client_id {client_id}"})
                    return

                self.state.uploads[client_id] = {
                    "client_id": client_id,
                    "categories": list(req["categories"]),
                    "patch_shapes": dict(req.get("patch_shapes", {})),
                    "memory_banks": dict(req["memory_banks"]),
                }
                send_msg(conn, {"ok": True, "uploaded": True})
                self._aggregate_if_ready()
                return

            if action == "status":
                send_msg(conn, {"ok": True, "global_ready": self.state.global_ready})
                return

            if action == "download":
                if not self.state.global_ready:
                    send_msg(conn, {"ok": False, "error": "global_not_ready"})
                    return
                categories = req.get("categories")
                if categories is None:
                    categories = sorted(self.state.global_banks.keys())
                out_banks = {cat: self.state.global_banks[cat] for cat in categories if cat in self.state.global_banks}
                out_shapes = {cat: self.state.patch_shapes.get(cat) for cat in out_banks.keys()}
                send_msg(conn, {"ok": True, "global_banks": out_banks, "patch_shapes": out_shapes})
                return

            if action == "shutdown":
                send_msg(conn, {"ok": True, "shutting_down": True})
                self._shutdown_event.set()
                return

            send_msg(conn, {"ok": False, "error": f"unknown action {action}"})

    def serve_forever(self) -> None:
        logging.info("[Server] Starting sequential server on %s:%d", self.host, self.port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(16)
            s.settimeout(1.0)

            while not self._shutdown_event.is_set():
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    self._handle_client(conn, addr)
                except Exception as e:
                    logging.exception("[Server] Error handling client %s: %s", addr, e)

        logging.info("[Server] Server stopped.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential Federated PatchCore Server")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--expected-client-ids", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--server-max-samples", type=int, default=5000)
    p.add_argument("--server-coreset-chunk-size", type=int, default=16384)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/federated_sequential/server"))
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("Starting Sequential Federated PatchCore Server")
    print(f"Expected clients: {args.expected_client_ids}")
    print(f"Address: {args.host}:{args.port}")
    print(f"Server k-center max_samples: {args.server_max_samples}")
    print("=" * 60)

    server = SequentialFederatedServer(
        host=args.host,
        port=args.port,
        expected_client_ids=args.expected_client_ids,
        server_max_samples=args.server_max_samples,
        server_coreset_chunk_size=args.server_coreset_chunk_size,
        output_dir=args.output_dir,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
