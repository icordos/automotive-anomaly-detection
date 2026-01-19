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
    pending_uploads: Dict[int, dict]
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
        enable_bank_audit: bool,
        bank_norm_min: Optional[float],
        bank_norm_max: Optional[float],
        bank_cosine_min: Optional[float],
        bank_max_outlier_fraction: float,
        bank_reject_on_audit: bool,
        bank_filter_outliers: bool,
        aggregation_method: str,
        aggregation_trim_fraction: float,
        aggregation_cosine_min: Optional[float],
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
            pending_uploads={},
            global_ready=False,
            global_banks={},
            patch_shapes={},
        )

        self._shutdown_event = threading.Event()
        self.enable_bank_audit = enable_bank_audit
        self.bank_norm_min = bank_norm_min
        self.bank_norm_max = bank_norm_max
        self.bank_cosine_min = bank_cosine_min
        self.bank_max_outlier_fraction = bank_max_outlier_fraction
        self.bank_reject_on_audit = bank_reject_on_audit
        self.bank_filter_outliers = bank_filter_outliers
        self.aggregation_method = aggregation_method
        self.aggregation_trim_fraction = aggregation_trim_fraction
        self.aggregation_cosine_min = aggregation_cosine_min
        if self.aggregation_method not in {"concat", "median", "trimmed_mean"}:
            raise ValueError(f"Unsupported aggregation_method {self.aggregation_method}")
        if not 0 <= self.aggregation_trim_fraction < 0.5:
            raise ValueError("aggregation_trim_fraction must be in [0, 0.5)")

    def _all_uploaded(self) -> bool:
        return all(cid in self.state.uploads for cid in self.state.expected_client_ids)

    def _trimmed_mean(self, values: np.ndarray, trim_fraction: float) -> np.ndarray:
        if values.shape[0] <= 2 or trim_fraction <= 0:
            return values.mean(axis=0)
        trim = int(values.shape[0] * trim_fraction)
        if trim == 0:
            return values.mean(axis=0)
        if 2 * trim >= values.shape[0]:
            return values.mean(axis=0)
        sorted_vals = np.sort(values, axis=0)
        trimmed = sorted_vals[trim:-trim]
        return trimmed.mean(axis=0)

    def _audit_memory_bank(
        self,
        bank: np.ndarray,
        category: str,
        client_id: int,
    ) -> Tuple[np.ndarray, bool]:
        if bank.size == 0:
            return bank, False

        finite_mask = np.isfinite(bank).all(axis=1)
        if not np.all(finite_mask):
            logging.warning(
                "[Server] %s client %d: found %d non-finite rows",
                category,
                client_id,
                int((~finite_mask).sum()),
            )

        mask = finite_mask.copy()
        norms = None
        if self.bank_norm_min is not None or self.bank_norm_max is not None:
            norms = np.linalg.norm(bank, axis=1)
            if self.bank_norm_min is not None:
                mask &= norms >= self.bank_norm_min
            if self.bank_norm_max is not None:
                mask &= norms <= self.bank_norm_max

        cosine_mean = float("nan")
        if self.bank_cosine_min is not None:
            centroid = bank.mean(axis=0)
            denom = np.linalg.norm(centroid)
            if denom > 0:
                cosine = (bank @ centroid) / (np.linalg.norm(bank, axis=1) * denom + 1e-12)
                cosine_mean = float(np.mean(cosine))
                mask &= cosine >= self.bank_cosine_min

        filtered = bank[mask]
        outlier_fraction = 1.0 - (filtered.shape[0] / max(1, bank.shape[0]))

        if self.enable_bank_audit:
            logging.info(
                "[Server] %s client %d: audit rows=%d kept=%d outliers=%.2f norms_mean=%s cosine_mean=%s",
                category,
                client_id,
                bank.shape[0],
                filtered.shape[0],
                outlier_fraction,
                "na" if norms is None else f"{float(np.mean(norms)):.4f}",
                "na" if np.isnan(cosine_mean) else f"{cosine_mean:.4f}",
            )

        failed = outlier_fraction > self.bank_max_outlier_fraction or filtered.size == 0
        if self.bank_reject_on_audit and failed:
            return bank, True

        if self.bank_filter_outliers:
            if filtered.size == 0:
                return bank, False
            return filtered, False

        return bank, False

    def _aggregate_if_ready(self) -> None:
        if self.state.global_ready:
            return
        if not self._all_uploaded():
            return

        logging.info("[Server] All clients uploaded. Aggregating global memory banks...")

        category_to_banks: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        patch_shapes: Dict[str, Tuple[int, int]] = {}

        for cid, payload in self.state.uploads.items():
            categories = payload["categories"]
            client_patch_shapes = payload.get("patch_shapes", {})
            banks = payload["memory_banks"]
            for cat in categories:
                mb = banks.get(cat)
                if mb is None or mb.size == 0:
                    continue
                category_to_banks.setdefault(cat, []).append((cid, mb))
                if cat in client_patch_shapes:
                    patch_shapes[cat] = tuple(client_patch_shapes[cat])

        global_banks: Dict[str, np.ndarray] = {}
        for cat, mb_list in category_to_banks.items():
            client_ids = [cid for cid, _ in mb_list]
            banks_only = [mb for _, mb in mb_list]
            concat = np.concatenate(banks_only, axis=0)
            if self.aggregation_method in {"median", "trimmed_mean"}:
                centroids = np.stack([mb.mean(axis=0) for mb in banks_only], axis=0)
                if self.aggregation_method == "median":
                    robust_centroid = np.median(centroids, axis=0)
                else:
                    robust_centroid = self._trimmed_mean(centroids, self.aggregation_trim_fraction)
                denom = np.linalg.norm(robust_centroid) + 1e-12
                cosines = (centroids @ robust_centroid) / (np.linalg.norm(centroids, axis=1) * denom + 1e-12)
                if self.aggregation_cosine_min is not None:
                    keep_mask = cosines >= self.aggregation_cosine_min
                    if not np.any(keep_mask):
                        logging.warning("[Server] %s: robust aggregation filtered all clients; using all.", cat)
                        keep_mask = np.ones_like(cosines, dtype=bool)
                    kept_ids = [cid for cid, keep in zip(client_ids, keep_mask) if keep]
                    kept_banks = [mb for mb, keep in zip(banks_only, keep_mask) if keep]
                    concat = np.concatenate(kept_banks, axis=0)
                    logging.info(
                        "[Server] %s: robust aggregation kept clients=%s (cosine_min=%s)",
                        cat,
                        kept_ids,
                        self.aggregation_cosine_min,
                    )
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

    def _init_chunked_upload(
        self,
        client_id: int,
        categories: List[str],
        patch_shapes: Dict[str, Tuple[int, int]],
        expected_chunks: Dict[str, int],
    ) -> None:
        self.state.pending_uploads[client_id] = {
            "client_id": client_id,
            "categories": categories,
            "patch_shapes": patch_shapes,
            "expected_chunks": expected_chunks,
            "received_chunks": {cat: set() for cat in categories},
            "chunks": {cat: {} for cat in categories},
        }

    def _record_chunk(
        self,
        client_id: int,
        category: str,
        chunk_idx: int,
        total_chunks: int,
        data: np.ndarray,
    ) -> None:
        pending = self.state.pending_uploads.get(client_id)
        if pending is None:
            raise RuntimeError("upload_init must be called before upload_chunk")
        expected = pending["expected_chunks"].get(category)
        if expected is None:
            raise RuntimeError(f"Unexpected category {category}")
        if expected != total_chunks:
            raise RuntimeError(f"Total chunks mismatch for {category}: {expected} vs {total_chunks}")
        if not (0 <= chunk_idx < total_chunks):
            raise RuntimeError(f"Invalid chunk_idx {chunk_idx} for {category}")
        pending["chunks"][category][chunk_idx] = data
        pending["received_chunks"][category].add(chunk_idx)

    def _finalize_chunked_upload(self, client_id: int) -> dict:
        pending = self.state.pending_uploads.get(client_id)
        if pending is None:
            return {"ok": False, "error": "no_pending_upload"}

        for cat in pending["categories"]:
            expected = pending["expected_chunks"].get(cat, 0)
            received = pending["received_chunks"].get(cat, set())
            if len(received) != expected:
                return {
                    "ok": False,
                    "error": f"missing_chunks:{cat}:{len(received)}/{expected}",
                }

        memory_banks: Dict[str, np.ndarray] = {}
        for cat in pending["categories"]:
            expected = pending["expected_chunks"].get(cat, 0)
            if expected == 0:
                memory_banks[cat] = np.empty((0, 0), dtype=np.float32)
                continue
            chunks = pending["chunks"][cat]
            ordered = [chunks[idx] for idx in range(expected)]
            memory_banks[cat] = np.concatenate(ordered, axis=0)

        if self.enable_bank_audit:
            audited_banks: Dict[str, np.ndarray] = {}
            for cat in pending["categories"]:
                mb = memory_banks.get(cat)
                if mb is None:
                    continue
                filtered, failed = self._audit_memory_bank(mb, cat, client_id)
                if failed:
                    return {"ok": False, "error": f"audit_failed:{cat}"}
                audited_banks[cat] = filtered
            memory_banks = audited_banks

        self.state.uploads[client_id] = {
            "client_id": client_id,
            "categories": list(pending["categories"]),
            "patch_shapes": dict(pending.get("patch_shapes", {})),
            "memory_banks": memory_banks,
        }
        self.state.pending_uploads.pop(client_id, None)
        self._aggregate_if_ready()
        return {"ok": True, "uploaded": True}

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

                memory_banks = dict(req["memory_banks"])
                if self.enable_bank_audit:
                    audited_banks: Dict[str, np.ndarray] = {}
                    for cat in req["categories"]:
                        mb = memory_banks.get(cat)
                        if mb is None:
                            continue
                        filtered, failed = self._audit_memory_bank(mb, cat, client_id)
                        if failed:
                            send_msg(conn, {"ok": False, "error": f"audit_failed:{cat}"})
                            return
                        audited_banks[cat] = filtered
                    memory_banks = audited_banks

                self.state.uploads[client_id] = {
                    "client_id": client_id,
                    "categories": list(req["categories"]),
                    "patch_shapes": dict(req.get("patch_shapes", {})),
                    "memory_banks": memory_banks,
                }
                send_msg(conn, {"ok": True, "uploaded": True})
                self._aggregate_if_ready()
                return
            if action == "upload_init":
                client_id = int(req["client_id"])
                logging.info("[Server] upload_init from client_id=%s addr=%s", client_id, addr)

                if client_id not in self.state.expected_client_ids:
                    send_msg(conn, {"ok": False, "error": f"unexpected client_id {client_id}"})
                    return

                categories = list(req["categories"])
                patch_shapes = dict(req.get("patch_shapes", {}))
                expected_chunks = dict(req.get("expected_chunks", {}))
                self._init_chunked_upload(client_id, categories, patch_shapes, expected_chunks)
                send_msg(conn, {"ok": True, "init": True})
                return

            if action == "upload_chunk":
                client_id = int(req["client_id"])
                category = req["category"]
                chunk_idx = int(req["chunk_idx"])
                total_chunks = int(req["total_chunks"])
                data = req["data"]
                try:
                    self._record_chunk(client_id, category, chunk_idx, total_chunks, data)
                except Exception as exc:
                    send_msg(conn, {"ok": False, "error": str(exc)})
                    return
                send_msg(conn, {"ok": True, "received": True})
                return

            if action == "upload_complete":
                client_id = int(req["client_id"])
                logging.info("[Server] upload_complete from client_id=%s addr=%s", client_id, addr)
                resp = self._finalize_chunked_upload(client_id)
                send_msg(conn, resp)
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
    p.add_argument("--enable-bank-audit", action="store_true")
    p.add_argument("--bank-norm-min", type=float, default=None)
    p.add_argument("--bank-norm-max", type=float, default=None)
    p.add_argument("--bank-cosine-min", type=float, default=None)
    p.add_argument("--bank-max-outlier-fraction", type=float, default=0.2)
    p.add_argument("--bank-reject-on-audit", action="store_true")
    p.add_argument("--bank-filter-outliers", action="store_true")
    p.add_argument(
        "--aggregation-method",
        type=str,
        default="concat",
        choices=["concat", "median", "trimmed_mean"],
    )
    p.add_argument("--aggregation-trim-fraction", type=float, default=0.1)
    p.add_argument("--aggregation-cosine-min", type=float, default=None)
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

    enable_bank_audit = (
        args.enable_bank_audit
        or args.bank_norm_min is not None
        or args.bank_norm_max is not None
        or args.bank_cosine_min is not None
        or args.bank_filter_outliers
        or args.bank_reject_on_audit
    )

    server = SequentialFederatedServer(
        host=args.host,
        port=args.port,
        expected_client_ids=args.expected_client_ids,
        server_max_samples=args.server_max_samples,
        server_coreset_chunk_size=args.server_coreset_chunk_size,
        output_dir=args.output_dir,
        enable_bank_audit=enable_bank_audit,
        bank_norm_min=args.bank_norm_min,
        bank_norm_max=args.bank_norm_max,
        bank_cosine_min=args.bank_cosine_min,
        bank_max_outlier_fraction=args.bank_max_outlier_fraction,
        bank_reject_on_audit=args.bank_reject_on_audit,
        bank_filter_outliers=args.bank_filter_outliers,
        aggregation_method=args.aggregation_method,
        aggregation_trim_fraction=args.aggregation_trim_fraction,
        aggregation_cosine_min=args.aggregation_cosine_min,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
