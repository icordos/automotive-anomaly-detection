#!/usr/bin/env python
"""Sequential Federated PatchCore Client (no Flower).

Modes:
- upload: build local memory banks for categories and upload to server
- eval: download global banks for categories from server, evaluate locally, write metrics json
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from patchcore_training import PatchCoreTrainer, PatchCoreTrainingConfig


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


def rpc(server_host: str, server_port: int, payload: dict, timeout_s: float = 120.0) -> dict:
    with socket.create_connection((server_host, server_port), timeout=timeout_s) as sock:
        send_msg(sock, payload)
        return recv_msg(sock)


def _rdp_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float,
    orders: List[float],
) -> Tuple[float, float]:
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0 for RDP accounting.")
    if steps <= 0:
        return 0.0, orders[0]
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1) for RDP accounting.")

    rdp = [(steps * order) / (2.0 * noise_multiplier ** 2) for order in orders]
    epsilons = [
        r + (np.log(1.0 / delta) / (order - 1.0))
        for r, order in zip(rdp, orders)
    ]
    min_idx = int(np.argmin(epsilons))
    return float(epsilons[min_idx]), float(orders[min_idx])


class SequentialPatchCoreClient:
    def __init__(
        self,
        client_id: int,
        categories: List[str],
        config: PatchCoreTrainingConfig,
        output_dir: Path,
        dp_enabled: bool,
        dp_epsilon: float,
        dp_delta: float,
        dp_clip_norm: float,
        dp_seed: int | None,
    ) -> None:
        self.client_id = client_id
        self.categories = categories
        self.config = config
        self.output_dir = output_dir
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip_norm = dp_clip_norm
        self.dp_seed = dp_seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trainer = PatchCoreTrainer(config)
        self.local_memory_banks: Dict[str, torch.Tensor] = {}
        self.patch_shapes: Dict[str, Tuple[int, int]] = {}

    def build_local_banks(self) -> None:
        logging.info("[Client %d] Building local memory banks for categories: %s", self.client_id, self.categories)
        for category in self.categories:
            logging.info("[Client %d] fit(%s) start", self.client_id, category)
            checkpoint_path, memory_bank, patch_shape = self.trainer.fit(category)
            self.local_memory_banks[category] = memory_bank
            self.patch_shapes[category] = patch_shape
            logging.info(
                "[Client %d] fit(%s) done: bank=%s checkpoint=%s",
                self.client_id,
                category,
                tuple(memory_bank.shape),
                checkpoint_path,
            )
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def _apply_dp(self, banks_np: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.dp_enabled:
            return banks_np
        if self.dp_epsilon <= 0:
            raise ValueError("dp_epsilon must be > 0 when DP is enabled.")
        if self.dp_delta <= 0 or self.dp_delta >= 1:
            raise ValueError("dp_delta must be in (0, 1) when DP is enabled.")
        if self.dp_clip_norm <= 0:
            raise ValueError("dp_clip_norm must be > 0 when DP is enabled.")

        sigma = float(np.sqrt(2 * np.log(1.25 / self.dp_delta)) / self.dp_epsilon)
        noise_std = sigma * self.dp_clip_norm
        rng = np.random.default_rng(self.dp_seed)

        dp_banks: Dict[str, np.ndarray] = {}
        for cat, mb in banks_np.items():
            if mb.size == 0:
                dp_banks[cat] = mb
                continue
            norms = np.linalg.norm(mb, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.dp_clip_norm / (norms + 1e-12))
            clipped = mb * scale
            noise = rng.normal(0.0, noise_std, size=clipped.shape).astype(clipped.dtype, copy=False)
            dp_banks[cat] = clipped + noise

        steps = max(1, len(banks_np))
        orders = [1.25, 1.5, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128]
        rdp_eps, rdp_order = _rdp_epsilon(sigma, steps, self.dp_delta, orders)
        logging.info(
            "[Client %d] DP enabled: per_release_eps=%.4f delta=%.2e clip=%.3f sigma=%.4f rdp_eps=%.4f (order=%.2f, steps=%d)",
            self.client_id,
            self.dp_epsilon,
            self.dp_delta,
            self.dp_clip_norm,
            sigma,
            rdp_eps,
            rdp_order,
            steps,
        )
        budget_path = self.output_dir / f"client_{self.client_id}_dp_budget.json"
        budget_payload = {
            "client_id": self.client_id,
            "per_release_eps": float(self.dp_epsilon),
            "delta": float(self.dp_delta),
            "clip_norm": float(self.dp_clip_norm),
            "sigma": float(sigma),
            "steps": int(steps),
            "rdp_eps": float(rdp_eps),
            "rdp_order": float(rdp_order),
        }
        with open(budget_path, "w", encoding="utf-8") as f:
            json.dump(budget_payload, f, indent=2)
        logging.info("[Client %d] Wrote DP budget to %s", self.client_id, budget_path)
        return dp_banks

    def upload(self, server_host: str, server_port: int) -> None:
        banks_np: Dict[str, np.ndarray] = {}
        for cat, mb in self.local_memory_banks.items():
            banks_np[cat] = mb.cpu().numpy().astype(np.float32, copy=False)
        banks_np = self._apply_dp(banks_np)

        payload = {
            "action": "upload",
            "client_id": self.client_id,
            "categories": self.categories,
            "patch_shapes": self.patch_shapes,
            "memory_banks": banks_np,
        }
        resp = rpc(server_host, server_port, payload, timeout_s=300.0)
        if not resp.get("ok"):
            raise RuntimeError(f"Upload failed: {resp}")
        logging.info("[Client %d] Upload OK", self.client_id)

    def download_global(self, server_host: str, server_port: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[int, int]]]:
        resp = rpc(
            server_host,
            server_port,
            {"action": "download", "categories": self.categories},
            timeout_s=300.0,
        )
        if not resp.get("ok"):
            raise RuntimeError(f"Download failed: {resp}")
        banks = resp["global_banks"]
        shapes = resp.get("patch_shapes", {}) or {}
        banks_t: Dict[str, torch.Tensor] = {}
        shapes_t: Dict[str, Tuple[int, int]] = {}
        for cat, arr in banks.items():
            banks_t[cat] = torch.from_numpy(arr).to(self.config.resolved_device())
            if shapes.get(cat) is not None:
                shapes_t[cat] = tuple(shapes[cat])
        return banks_t, shapes_t

    def evaluate_with_global(self, server_host: str, server_port: int) -> Dict[str, float]:
        global_banks, _ = self.download_global(server_host, server_port)

        all_metrics: Dict[str, float] = {}
        valid_aurocs: List[float] = []
        total_test_images = 0

        for cat in self.categories:
            if cat not in global_banks:
                logging.warning("[Client %d] Missing global bank for %s", self.client_id, cat)
                continue

            self.trainer.load_memory_bank(global_banks[cat].cpu())
            metrics_dict, n_test = self.trainer.evaluate(cat)
            total_test_images += n_test

            for k, v in metrics_dict.items():
                all_metrics[f"{cat}_{k}"] = float(v)

            au = metrics_dict.get("image_roc_auc")
            if au is not None and not np.isnan(au):
                valid_aurocs.append(float(au))

            torch.cuda.empty_cache()
            import gc
            gc.collect()

        avg_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
        all_metrics["avg_auroc"] = avg_auroc
        all_metrics["total_test_images"] = float(total_test_images)

        out_path = self.output_dir / f"client_{self.client_id}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        logging.info("[Client %d] Wrote metrics to %s", self.client_id, out_path)

        return all_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential Federated PatchCore Client")
    p.add_argument("--mode", type=str, choices=["upload", "eval"], required=True)

    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--server-host", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8081)

    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/federated_sequential/clients"))

    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--coreset-method", type=str, default="random", choices=["random", "kcenter"])
    p.add_argument("--coreset-max-samples", type=int, default=5000)
    p.add_argument("--coreset-chunk-size", type=int, default=16384)
    p.add_argument("--distance-chunk-size", type=int, default=8192)
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--train-partition-id", type=int, default=0)
    p.add_argument("--train-num-partitions", type=int, default=1)
    p.add_argument("--dp", action="store_true", help="Enable client-side DP before upload")
    p.add_argument("--dp-epsilon", type=float, default=1.0)
    p.add_argument("--dp-delta", type=float, default=1e-5)
    p.add_argument("--dp-clip-norm", type=float, default=1.0)
    p.add_argument("--dp-seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(message)s")

    client_out_dir = args.output_dir / f"client_{args.client_id}"
    client_out_dir.mkdir(parents=True, exist_ok=True)

    config = PatchCoreTrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=client_out_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        coreset_sampling_ratio=args.coreset_ratio,
        coreset_method=args.coreset_method,
        coreset_max_samples=args.coreset_max_samples,
        coreset_chunk_size=args.coreset_chunk_size,
        distance_chunk_size=args.distance_chunk_size,
        use_mixed_precision=not args.no_mixed_precision,
        device=args.device,
        train_partition_id=args.train_partition_id,
        train_num_partitions=args.train_num_partitions,
    )

    client = SequentialPatchCoreClient(
        client_id=args.client_id,
        categories=args.categories,
        config=config,
        output_dir=client_out_dir,
        dp_enabled=args.dp,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_clip_norm=args.dp_clip_norm,
        dp_seed=args.dp_seed,
    )

    if args.mode == "upload":
        client.build_local_banks()
        client.upload(args.server_host, args.server_port)
        return

    if args.mode == "eval":
        client.evaluate_with_global(args.server_host, args.server_port)
        return


if __name__ == "__main__":
    main()
