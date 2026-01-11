#!/usr/bin/env python
"""PatchCore training loop for AutoVI datasets.

Builds a memory bank of nominal patch features for each category and supports
coreset sampling (random or k-center greedy). Also provides evaluation utilities.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def log_gpu_memory(prefix: str = "") -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.debug(
            f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


@dataclass
class PatchCoreTrainingConfig:
    dataset_root: Path
    output_dir: Path
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 2
    device: Optional[str] = None
    feature_layers: Tuple[str, ...] = ("layer2", "layer3")
    coreset_sampling_ratio: float = 0.1
    coreset_method: str = "kcenter"
    coreset_max_samples: Optional[int] = None
    coreset_chunk_size: int = 8192
    use_mixed_precision: bool = True
    distance_chunk_size: int = 8192
    train_partition_id: int = 0
    train_num_partitions: int = 1

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if not 0.0 <= self.coreset_sampling_ratio <= 1.0:
            raise ValueError(
                f"coreset_sampling_ratio must be in [0, 1], got {self.coreset_sampling_ratio}"
            )
        if self.coreset_max_samples is not None and self.coreset_max_samples <= 0:
            raise ValueError(
                f"coreset_max_samples must be positive, got {self.coreset_max_samples}"
            )
        if self.coreset_chunk_size <= 0:
            raise ValueError(
                f"coreset_chunk_size must be positive, got {self.coreset_chunk_size}"
            )
        if self.distance_chunk_size <= 0:
            raise ValueError(
                f"distance_chunk_size must be positive, got {self.distance_chunk_size}"
            )
        if self.coreset_method.lower() not in {"kcenter", "farthest", "random"}:
            raise ValueError(
                f"coreset_method must be 'kcenter', 'farthest', or 'random', got {self.coreset_method}"
            )
        if self.train_num_partitions <= 0:
            raise ValueError(f"train_num_partitions must be positive, got {self.train_num_partitions}")
        if not (0 <= self.train_partition_id < self.train_num_partitions):
            raise ValueError(
                f"train_partition_id must be in [0, train_num_partitions-1], got {self.train_partition_id} for {self.train_num_partitions}"
            )

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


class AutoVIAnomalyDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        category: str,
        split: str,
        image_transform: transforms.Compose,
        mask_transform: transforms.Compose,
        train_partition_id: int = 0,
        train_num_partitions: int = 1,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split '{split}'")
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.train_partition_id = train_partition_id
        self.train_num_partitions = train_num_partitions
        category_dir = dataset_root / category / category
        if not category_dir.exists():
            raise FileNotFoundError(f"Missing category directory: {category_dir}")
        self.items: List[Tuple[Path, Optional[List[Path]], int]] = []
        self._collect_items(category_dir)

    def _collect_items(self, category_dir: Path) -> None:
        if self.split == "train":
            train_dir = category_dir / "train" / "good"
            all_images = sorted(train_dir.glob("*.png"))

            # Optional deterministic partitioning (federated split)
            train_partition_id = getattr(self, "train_partition_id", 0)
            train_num_partitions = getattr(self, "train_num_partitions", 1)

            if train_num_partitions > 1:
                import hashlib

                def _keep(p: Path) -> bool:
                    h = hashlib.md5(str(p.name).encode("utf-8")).hexdigest()
                    return (int(h, 16) % train_num_partitions) == train_partition_id

                all_images = [p for p in all_images if _keep(p)]

            for image_path in all_images:
                self.items.append((image_path, None, 0))
            return

        test_dir = category_dir / "test"
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            label = 0 if defect_dir.name == "good" else 1
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_paths: Optional[List[Path]] = None
                if label == 1:
                    ground_truth_dir = (
                        category_dir / "ground_truth" / defect_dir.name / image_path.stem
                    )
                    if ground_truth_dir.exists():
                        mask_paths = sorted(ground_truth_dir.glob("*.png"))
                self.items.append((image_path, mask_paths, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        image_path, mask_paths, label = self.items[index]

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            raise RuntimeError(f"Cannot load image {image_path}") from e

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.split == "train" or mask_paths is None:
            mask_tensor = torch.zeros(
                (1, image_tensor.shape[-2], image_tensor.shape[-1]), dtype=torch.float32
            )
        else:
            masks = []
            for mask_path in mask_paths:
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    masks.append(self.mask_transform(mask_img))
                except Exception as e:
                    logging.warning(f"Failed to load mask {mask_path}: {e}. Skipping.")
                    continue
            if masks:
                mask_tensor = torch.clamp(
                    torch.sum(torch.stack(masks), dim=0), min=0.0, max=1.0
                )
            else:
                mask_tensor = torch.zeros(
                    (1, image_tensor.shape[-2], image_tensor.shape[-1]), dtype=torch.float32
                )
        return image_tensor, label_tensor, mask_tensor


class PatchCoreModel:
    def __init__(self, config: PatchCoreTrainingConfig) -> None:
        self.layers = config.feature_layers
        self.device = torch.device(config.resolved_device())
        self.use_mixed_precision = config.use_mixed_precision and self.device.type == "cuda"
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        backbone = wide_resnet50_2(weights=weights)
        backbone.eval()
        self.feature_extractor = create_feature_extractor(
            backbone, return_nodes={layer: layer for layer in self.layers}
        ).to(self.device)
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad_(False)

    def extract(self, batch: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    features = self.feature_extractor(batch)
                return {k: v.float() for k, v in features.items()}
            return self.feature_extractor(batch)

    def aggregate(self, features: Dict[str, Tensor]) -> Tuple[Tensor, Tuple[int, int]]:
        patch_maps = []
        target_hw: Optional[Tuple[int, int]] = None
        for layer in self.layers:
            fmap = features[layer]
            if target_hw is None:
                target_hw = (fmap.shape[-2], fmap.shape[-1])
            if fmap.shape[-2:] != target_hw:
                fmap = F.interpolate(
                    fmap,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            patch_maps.append(fmap)
        combined = torch.cat(patch_maps, dim=1)
        bsz, channels, height, width = combined.shape
        embeddings = combined.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        return embeddings, (height, width)


class PatchCoreTrainer:
    def __init__(self, config: PatchCoreTrainingConfig) -> None:
        self.config = config
        self.model = PatchCoreModel(config)
        self.memory_bank: Optional[Tensor] = None
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def _loader(self, category: str, split: str) -> DataLoader:
        dataset = AutoVIAnomalyDataset(
            dataset_root=self.config.dataset_root,
            category=category,
            split=split,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            train_partition_id=self.config.train_partition_id,
            train_num_partitions=self.config.train_num_partitions,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def fit(self, category: str) -> Tuple[Path, Tensor, Optional[Tuple[int, int]]]:
        loader = self._loader(category, "train")
        logging.info("Starting feature extraction for %s (images=%d)", category, len(loader.dataset))
        log_gpu_memory("[Before training] ")
        start_time = time.perf_counter()
        feature_chunks: List[Tensor] = []
        patch_shape: Optional[Tuple[int, int]] = None
        for images, _, _ in tqdm(loader, desc=f"Extracting {category} features"):
            images = images.to(self.model.device)
            feats = self.model.extract(images)
            embeddings, patch_shape = self.model.aggregate(feats)
            feature_chunks.append(embeddings.reshape(-1, embeddings.shape[-1]).cpu())
        if not feature_chunks:
            raise RuntimeError(f"No training data found for category '{category}'")

        extraction_time = time.perf_counter() - start_time
        memory_bank = torch.cat(feature_chunks, dim=0)
        logging.info(
            "Finished feature extraction for %s (patches=%d, dim=%d) in %.2fs (%.2f images/sec)",
            category,
            memory_bank.shape[0],
            memory_bank.shape[1],
            extraction_time,
            len(loader.dataset) / extraction_time,
        )
        log_gpu_memory("[After extraction] ")

        memory_bank = self._apply_coreset(memory_bank)
        self.memory_bank = memory_bank.to(self.model.device)
        checkpoint_path = self._save_checkpoint(category, memory_bank, patch_shape)
        logging.info(
            "Saved %s checkpoint with %d features to %s",
            category,
            self.memory_bank.shape[0],
            checkpoint_path,
        )
        log_gpu_memory("[After training] ")
        return checkpoint_path, memory_bank, patch_shape

    def load_memory_bank(self, memory_bank: Tensor) -> None:
        self.memory_bank = memory_bank.to(self.model.device)
        logging.info("Loaded memory bank with %d features", self.memory_bank.shape[0])

    def _apply_coreset(self, features: Tensor) -> Tensor:
        ratio = self.config.coreset_sampling_ratio
        if ratio >= 1.0:
            logging.info("Skipping coreset sampling; ratio %.2f", ratio)
            return features
        total = features.shape[0]
        target = max(1, int(total * ratio))
        if self.config.coreset_max_samples is not None:
            target = min(target, self.config.coreset_max_samples)
        if target >= total:
            logging.info(
                "Coreset target (%d) >= total patches (%d); returning original features",
                target,
                total,
            )
            return features
        logging.info(
            "Applying %s coreset sampling (ratio=%.3f total=%d target=%d)",
            self.config.coreset_method,
            ratio,
            total,
            target,
        )
        start_time = time.perf_counter()
        method = self.config.coreset_method.lower()
        if method == "random":
            selected_indices = torch.randperm(total)[:target]
        elif method in {"kcenter", "farthest"}:
            selected_indices = self._kcenter_greedy(features, target)
        else:
            raise ValueError(
                f"Unsupported coreset method '{self.config.coreset_method}'. Use 'random' or 'kcenter'."
            )
        logging.info(
            "%s coreset sampling finished in %.2fs",
            self.config.coreset_method,
            time.perf_counter() - start_time,
        )
        return features[selected_indices]

    def _kcenter_greedy(self, features: Tensor, target: int) -> Tensor:
        device = features.device
        num_vectors = features.shape[0]
        chunk_size = max(1, self.config.coreset_chunk_size)
        indices = torch.empty(target, dtype=torch.long)
        indices[0] = torch.randint(0, num_vectors, (1,), device=device).item()
        min_distances = torch.full((num_vectors,), float("inf"), device=device)
        progress = tqdm(range(1, target), desc="Coreset k-center", leave=False)
        for i in progress:
            last_feature = features[indices[i - 1]].unsqueeze(0)
            for start in range(0, num_vectors, chunk_size):
                end = min(start + chunk_size, num_vectors)
                chunk = features[start:end]
                distances = torch.cdist(chunk, last_feature).squeeze(1)
                min_distances[start:end] = torch.minimum(min_distances[start:end], distances)
            next_index = torch.argmax(min_distances).item()
            indices[i] = next_index
            if i % 100 == 0 or i == target - 1:
                progress.set_postfix(
                    latest_distance=float(min_distances[next_index]),
                    selected=i + 1,
                )
        return indices

    def _save_checkpoint(
        self,
        category: str,
        memory_bank: Tensor,
        patch_shape: Optional[Tuple[int, int]],
    ) -> Path:
        checkpoint = {
            "category": category,
            "memory_bank": memory_bank,
            "patch_shape": patch_shape,
            "image_size": self.config.image_size,
            "layers": self.config.feature_layers,
        }
        path = self.output_dir / f"{category}_patchcore.pt"
        torch.save(checkpoint, path)
        return path

    def evaluate(self, category: str) -> Tuple[Dict[str, float], int]:
        if self.memory_bank is None:
            raise RuntimeError("Call fit() or load_memory_bank() before evaluate().")
        loader = self._loader(category, "test")
        logging.info("Evaluating %s on %d test images", category, len(loader.dataset))
        log_gpu_memory("[Before evaluation] ")
        start_time = time.perf_counter()

        image_scores: List[float] = []
        image_labels: List[int] = []
        pixel_scores: List[float] = []
        pixel_labels: List[int] = []

        for i, (images, labels, masks) in enumerate(tqdm(loader, desc=f"Evaluating {category}")):
            images = images.to(self.model.device)
            feats = self.model.extract(images)
            embeddings, patch_shape = self.model.aggregate(feats)
            anomaly_scores, anomaly_maps = self._compute_scores(embeddings, patch_shape)
            image_scores.extend(anomaly_scores.cpu().tolist())
            image_labels.extend(labels.cpu().tolist())
            upsampled_maps = F.interpolate(
                anomaly_maps,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for upsampled, mask in zip(upsampled_maps, masks):
                pixel_scores.extend(upsampled.flatten().cpu().tolist())
                pixel_labels.extend(mask.flatten().long().cpu().tolist())

            del feats, embeddings, anomaly_scores, anomaly_maps, upsampled_maps, images
            if i % 10 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        eval_time = time.perf_counter() - start_time
        logging.info(
            "Evaluation completed in %.2fs (%.2f images/sec)",
            eval_time,
            len(loader.dataset) / eval_time,
        )
        log_gpu_memory("[After evaluation] ")

        metrics_dict = {
            "image_roc_auc": self._safe_auc(image_labels, image_scores),
            "pixel_roc_auc": self._safe_auc(pixel_labels, pixel_scores),
            "image_ap": self._safe_average_precision(image_labels, image_scores),
            "pixel_ap": self._safe_average_precision(pixel_labels, pixel_scores),
        }

        del image_scores, image_labels, pixel_scores, pixel_labels, loader
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logging.info("Memory cleaned after %s evaluation", category)

        return metrics_dict, self._count_test_images(category)

    def _count_test_images(self, category: str) -> int:
        ds = AutoVIAnomalyDataset(
            dataset_root=self.config.dataset_root,
            category=category,
            split="test",
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
        )
        return len(ds)

    def _compute_scores(
        self,
        embeddings: Tensor,
        patch_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank has not been computed yet.")
        bsz, num_patches, feat_dim = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, feat_dim).to(self.model.device)

        chunk_size = self.config.distance_chunk_size
        min_distances = torch.full(
            (flat_embeddings.shape[0],), float("inf"), device=self.model.device
        )

        for start in range(0, flat_embeddings.shape[0], chunk_size):
            end = min(start + chunk_size, flat_embeddings.shape[0])
            chunk = flat_embeddings[start:end]
            distances = torch.cdist(chunk, self.memory_bank)
            chunk_min_distances, _ = torch.min(distances, dim=1)
            min_distances[start:end] = chunk_min_distances

        patch_scores = min_distances.reshape(bsz, patch_shape[0], patch_shape[1])
        anomaly_maps = patch_scores.unsqueeze(1)
        k = max(1, int(0.01 * patch_scores.shape[-2] * patch_scores.shape[-1]))
        topk_vals = torch.topk(patch_scores.flatten(1), k=k, dim=1).values
        image_scores = topk_vals.mean(dim=1)
        return image_scores, anomaly_maps

    @staticmethod
    def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return float("nan")
        return float(metrics.roc_auc_score(labels, scores))

    @staticmethod
    def _safe_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return float("nan")
        return float(metrics.average_precision_score(labels, scores))


def discover_categories(dataset_root: Path) -> List[str]:
    return [p.name for p in sorted(dataset_root.iterdir()) if p.is_dir()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchCore trainer for AutoVI datasets")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/patchcore"))
    parser.add_argument("--categories", nargs="*", help="Specific categories to train (defaults to all detected)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument(
        "--coreset-method",
        type=str,
        default="kcenter",
        choices=["kcenter", "random"],
        help="Coreset sampling strategy (kcenter is precise but slow; random is fast)",
    )
    parser.add_argument("--coreset-max-samples", type=int, default=None)
    parser.add_argument("--coreset-chunk-size", type=int, default=8192)
    parser.add_argument("--distance-chunk-size", type=int, default=8192)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level_name = args.log_level.upper()
    log_level = getattr(logging, log_level_name, None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    categories: Optional[Sequence[str]] = args.categories if args.categories else None
    if categories is None:
        categories = discover_categories(dataset_root)
    if not categories:
        raise ValueError(f"No categories found in {dataset_root}")

    logging.info("Training categories: %s", ", ".join(categories))
    logging.info("Using device: %s", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    config = PatchCoreTrainingConfig(
        dataset_root=dataset_root,
        output_dir=args.output_dir,
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
    )
    trainer = PatchCoreTrainer(config)
    summary: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    for category in categories:
        checkpoint_path, _, _ = trainer.fit(category)
        metrics_dict = None
        if not args.skip_eval:
            metrics_dict, _ = trainer.evaluate(category)
            logging.info("%s metrics: %s", category, metrics_dict)
        summary[category] = {"checkpoint": str(checkpoint_path), "metrics": metrics_dict}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
