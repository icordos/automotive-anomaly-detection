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


class AddGaussianNoise:
    def __init__(self, std: float) -> None:
        self.std = float(std)

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


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
    save_interpretability: bool = False
    saliency_max_images: int = 10
    shap_max_images: int = 0
    shap_background: int = 20
    shap_max_patches: int = 64
    seg_masks: bool = True
    corruption_prob: float = 0.0
    corruption_strength: float = 0.2
    gaussian_noise_std: float = 0.0
    robust_norm_max: Optional[float] = None
    robust_cosine_min: Optional[float] = None
    anomaly_score_clip: Optional[float] = None

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
        if self.saliency_max_images < 0:
            raise ValueError(f"saliency_max_images must be >= 0, got {self.saliency_max_images}")
        if self.shap_max_images < 0:
            raise ValueError(f"shap_max_images must be >= 0, got {self.shap_max_images}")
        if self.shap_background <= 0:
            raise ValueError(f"shap_background must be > 0, got {self.shap_background}")
        if self.shap_max_patches <= 0:
            raise ValueError(f"shap_max_patches must be > 0, got {self.shap_max_patches}")
        if not 0.0 <= self.corruption_prob <= 1.0:
            raise ValueError(
                f"corruption_prob must be in [0, 1], got {self.corruption_prob}"
            )
        if self.corruption_strength < 0.0:
            raise ValueError(
                f"corruption_strength must be >= 0, got {self.corruption_strength}"
            )
        if self.gaussian_noise_std < 0.0:
            raise ValueError(
                f"gaussian_noise_std must be >= 0, got {self.gaussian_noise_std}"
            )
        if self.robust_norm_max is not None and self.robust_norm_max <= 0:
            raise ValueError(
                f"robust_norm_max must be positive, got {self.robust_norm_max}"
            )
        if self.robust_cosine_min is not None and not -1.0 <= self.robust_cosine_min <= 1.0:
            raise ValueError(
                f"robust_cosine_min must be in [-1, 1], got {self.robust_cosine_min}"
            )
        if self.anomaly_score_clip is not None and self.anomaly_score_clip <= 0:
            raise ValueError(
                f"anomaly_score_clip must be positive, got {self.anomaly_score_clip}"
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
        return_path: bool = False,
        seg_masks: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split '{split}'")
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.train_partition_id = train_partition_id
        self.train_num_partitions = train_num_partitions
        self.return_path = return_path
        self.seg_masks = seg_masks

        category_dir = self._resolve_category_dir(dataset_root, category)
        if not category_dir.exists():
            raise FileNotFoundError(f"Missing category directory: {category_dir}")

        self.items: List[Tuple[Path, Optional[List[Path]], int]] = []
        self._collect_items(category_dir)
    def _glob_images(self, d: Path) -> List[Path]:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        out: List[Path] = []
        for pat in exts:
            out.extend(d.glob(pat))
        return sorted(out)

    def _find_child_ci(self, base: Path, name: str) -> Path:
        p = base / name
        if p.exists():
            return p
        if base.exists():
            for c in base.iterdir():
                if c.is_dir() and c.name.lower() == name.lower():
                    return c
        return p  # may not exist
    @staticmethod
    def _resolve_category_dir(dataset_root: Path, category: str) -> Path:
        p1 = dataset_root / category / category
        if p1.exists():
            return p1
        p2 = dataset_root / category
        return p2

    @staticmethod
    def _label_from_dirname(dirname: str) -> Optional[int]:
        name = (dirname or "").strip().lower()
        if name in {"good", "ok"}:
            return 0
        if name in {"ko", "bad", "ng", "defect"}:
            return 1
        return None

    def _collect_items(self, category_dir: Path) -> None:
        if self.split == "train":
            train_root = category_dir / "train"
            if not train_root.exists():
                train_root = category_dir / "training"

            good_dir = self._find_child_ci(train_root, "good")
            ok_dir = self._find_child_ci(train_root, "ok")

            if good_dir.exists():
                all_images = self._glob_images(good_dir)
            elif ok_dir.exists():
                all_images = self._glob_images(ok_dir)
            else:
                all_images = []
                if train_root.exists():
                    for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                        all_images.extend(train_root.rglob(pat))
                all_images = sorted(all_images)

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
            label = self._label_from_dirname(defect_dir.name)
            if label is None:
                label = 0 if defect_dir.name == "good" else 1
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_paths: Optional[List[Path]] = None
                if self.seg_masks and label == 1:
                    ground_truth_dir = (
                        category_dir / "ground_truth" / defect_dir.name / image_path.stem
                    )
                    if ground_truth_dir.exists():
                        mask_paths = sorted(ground_truth_dir.glob("*.png"))
                self.items.append((image_path, mask_paths, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, str]:
        image_path, mask_paths, label = self.items[index]

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            raise RuntimeError(f"Cannot load image {image_path}") from e

        label_tensor = torch.tensor(label, dtype=torch.long)

        if (not self.seg_masks) or self.split == "train" or mask_paths is None:
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
        if self.return_path:
            return image_tensor, label_tensor, mask_tensor, str(image_path)
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
        self.train_image_transform = self._build_train_transform()
        self.eval_image_transform = self._build_eval_transform()
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def _build_train_transform(self) -> transforms.Compose:
        augmentations = []
        if self.config.corruption_prob > 0:
            strength = self.config.corruption_strength
            augmentations.append(
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4 * strength,
                            contrast=0.4 * strength,
                            saturation=0.4 * strength,
                            hue=0.1 * strength,
                        ),
                        transforms.RandomAutocontrast(),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.0 + strength),
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5 + 2 * strength)),
                    ],
                    p=self.config.corruption_prob,
                )
            )
        return transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size), interpolation=InterpolationMode.BICUBIC),
                *augmentations,
                transforms.ToTensor(),
                AddGaussianNoise(self.config.gaussian_noise_std),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _build_eval_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _loader(self, category: str, split: str) -> DataLoader:
        dataset = AutoVIAnomalyDataset(
            dataset_root=self.config.dataset_root,
            category=category,
            split=split,
            image_transform=self.train_image_transform if split == "train" else self.eval_image_transform,
            mask_transform=self.mask_transform,
            train_partition_id=self.config.train_partition_id,
            train_num_partitions=self.config.train_num_partitions,
            return_path=split == "test" and self.config.save_interpretability,
            seg_masks=self.config.seg_masks,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    # În patchcore_training.py, metoda fit():

    def fit(
        self, 
        category: str,
        fairness_mode: str = "none",
        subgroup_metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[Path, Tensor, Optional[Tuple[int, int]]]:
        """Fit PatchCore model on training data."""
        loader = self._loader(category, "train")
        logging.info("Starting feature extraction for %s (images=%d)", category, len(loader.dataset))
        log_gpu_memory("[Before training] ")
        start_time = time.perf_counter()
        
        feature_chunks: List[Tensor] = []
        patch_shape: Optional[Tuple[int, int]] = None
        image_paths: List[Path] = []
        patches_per_image: List[int] = []  # NEW: Track exact patch count per image
        
        # Collect dataset items for subgroup detection
        dataset = loader.dataset
        if hasattr(dataset, 'items'):
            image_paths = [item[0] for item in dataset.items]
        
        for images, _, _ in tqdm(loader, desc=f"Extracting {category} features"):
            images = images.to(self.model.device)
            feats = self.model.extract(images)
            embeddings, patch_shape = self.model.aggregate(feats)
            
            # Track how many patches each image produced
            batch_patches = embeddings.shape[0] * embeddings.shape[1]  # bsz * num_patches
            patches_per_image.extend([embeddings.shape[1]] * images.shape[0])
            
            feature_chunks.append(embeddings.reshape(-1, embeddings.shape[-1]))
        
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

        memory_bank = self._filter_adversarial_features(memory_bank)

        # Fairness-aware coreset sampling
        subgroup_labels = None
        if fairness_mode != "none" and image_paths:
            try:
                from fairness_utils import SubgroupDetector
                
                # Detect subgroups from paths
                subgroup_assignments = SubgroupDetector.detect_from_paths(
                    image_paths, category
                )
                
                # Create mapping: subgroup_name -> subgroup_id
                subgroup_name_to_id = {name: idx for idx, name in enumerate(subgroup_assignments.keys())}
                
                # Build per-patch subgroup labels
                subgroup_labels = []
                patch_offset = 0
                
                for img_idx, img_path in enumerate(image_paths):
                    # Find which subgroup this image belongs to
                    subgroup_id = 0  # Default
                    for sg_name, sg_image_indices in subgroup_assignments.items():
                        if img_idx in sg_image_indices:
                            subgroup_id = subgroup_name_to_id[sg_name]
                            break
                    
                    # Assign same subgroup to ALL patches from this image
                    num_patches_from_this_image = patches_per_image[img_idx]
                    subgroup_labels.extend([subgroup_id] * num_patches_from_this_image)
                
                subgroup_labels = np.array(subgroup_labels[:memory_bank.shape[0]])
                
                logging.info(
                    "Detected %d subgroups for %s (%d patches, %d images)",
                    len(subgroup_assignments),
                    category,
                    len(subgroup_labels),
                    len(image_paths),
                )
            
            except ImportError:
                logging.warning("fairness_utils not available. Using standard coreset.")
                fairness_mode = "none"
            except Exception as e:
                logging.exception("Subgroup detection failed: %s. Using standard coreset.", e)
                fairness_mode = "none"
        
        # Apply coreset sampling (fairness-aware or standard)
        if fairness_mode != "none" and subgroup_labels is not None:
            memory_bank = self.apply_fair_coreset(memory_bank, subgroup_labels, fairness_mode)
        else:
            if self.model.device.type == "cuda":
                memory_bank = memory_bank.to(self.model.device)
                memory_bank = self._apply_coreset(memory_bank)
            else:
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

    def _filter_adversarial_features(self, features: Tensor) -> Tensor:
        if self.config.robust_norm_max is None and self.config.robust_cosine_min is None:
            return features

        mask = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        if self.config.robust_norm_max is not None:
            norms = torch.linalg.norm(features, dim=1)
            mask &= norms <= float(self.config.robust_norm_max)

        if self.config.robust_cosine_min is not None:
            centroid = torch.mean(features, dim=0)
            denom = torch.linalg.norm(centroid) + 1e-12
            if float(denom) > 0:
                cosine = torch.matmul(features, centroid) / (torch.linalg.norm(features, dim=1) * denom + 1e-12)
                mask &= cosine >= float(self.config.robust_cosine_min)

        kept = int(mask.sum().item())
        if kept == 0:
            logging.warning("Robust filtering removed all features; keeping original bank.")
            return features
        if kept < features.shape[0]:
            logging.info(
                "Robust filtering kept %d/%d features (norm_max=%s cosine_min=%s)",
                kept,
                features.shape[0],
                self.config.robust_norm_max,
                self.config.robust_cosine_min,
            )
        return features[mask]

    def load_memory_bank(self, memory_bank: Tensor) -> None:
        self.memory_bank = memory_bank.to(self.model.device)
        logging.info("Loaded memory bank with %d features", self.memory_bank.shape[0])

    def apply_fair_coreset(
        self,
        features: Tensor,
        subgroup_labels: Optional[np.ndarray] = None,
        fairness_mode: str = "none",
    ) -> Tensor:
        """Apply coreset sampling with optional fairness constraints.
        
        Args:
            features: (N, D) feature tensor
            subgroup_labels: (N,) subgroup assignments (optional)
            fairness_mode: 'none', 'proportional', 'equal'
        
        Returns:
            Tensor: Selected features after fairness-aware coreset sampling
        """
        if fairness_mode == "none" or subgroup_labels is None:
            return self._apply_coreset(features)
        
        # Import fairness utilities
        try:
            from fairness_utils import FairCoreset
        except ImportError:
            logging.warning(
                "fairness_utils not found. Falling back to standard coreset sampling."
            )
            return self._apply_coreset(features)
        
        ratio = self.config.coreset_sampling_ratio
        total = features.shape[0]
        target = max(1, int(total * ratio))
        
        if self.config.coreset_max_samples is not None:
            target = min(target, self.config.coreset_max_samples)
        
        if target >= total:
            logging.info(
                "Fairness coreset target (%d) >= total (%d); returning original features",
                target,
                total,
            )
            return features
        
        logging.info(
            "Applying fairness-aware coreset (%s): %d -> %d",
            fairness_mode,
            total,
            target,
        )
        
        start_time = time.perf_counter()
        
        # Convert to CPU for fairness sampling
        features_cpu = features.cpu()
        
        indices = FairCoreset.balanced_sampling(
            features_cpu, subgroup_labels, target, method=fairness_mode
        )
        
        logging.info(
            "Fairness-aware coreset sampling (%s) finished in %.2fs",
            fairness_mode,
            time.perf_counter() - start_time,
        )
        
        return features[indices]

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

    def _unnormalize_image(self, image_tensor: Tensor) -> np.ndarray:
        image = image_tensor.detach().cpu().float()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0.0, 1.0)
        image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return image

    def _save_heatmap_overlay(self, image_tensor: Tensor, heatmap: np.ndarray, out_path: Path) -> None:
        img = self._unnormalize_image(image_tensor)
        if heatmap.ndim == 3:
            heatmap = heatmap.squeeze()
        heatmap = np.clip(heatmap, 0.0, 1.0)
        heat = (heatmap * 255).astype(np.uint8)
        overlay = np.zeros_like(img)
        overlay[..., 0] = heat
        alpha = 0.45
        blended = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
        Image.fromarray(blended).save(out_path)

    def _compute_gradcam(self, image_tensor: Tensor) -> np.ndarray:
        self.model.feature_extractor.zero_grad(set_to_none=True)
        image_tensor = image_tensor.unsqueeze(0).to(self.model.device)
        image_tensor.requires_grad_(True)
        with torch.set_grad_enabled(True):
            feats = self.model.feature_extractor(image_tensor)
            feats = {k: v.float() for k, v in feats.items()}
            target_layer = self.config.feature_layers[-1]
            feats[target_layer].requires_grad_(True)
            feats[target_layer].retain_grad()
            embeddings, patch_shape = self.model.aggregate(feats)
            image_scores, _ = self._compute_scores(embeddings, patch_shape)
            image_scores[0].backward()
            grads = feats[target_layer].grad
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feats[target_layer]).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        return cam

    def _save_interpretability_outputs(
        self,
        category: str,
        image_tensor: Tensor,
        image_path: str,
        anomaly_map: Tensor,
        patch_shape: Tuple[int, int],
    ) -> None:
        category_dir = self.output_dir / "interpretability" / category
        category_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        anomaly_np = anomaly_map.squeeze(0).detach().cpu().numpy()
        anomaly_np = (anomaly_np - anomaly_np.min()) / (anomaly_np.max() - anomaly_np.min() + 1e-12)
        anomaly_out = category_dir / f"{stem}_anomaly.png"
        self._save_heatmap_overlay(image_tensor, anomaly_np, anomaly_out)

        gradcam = self._compute_gradcam(image_tensor)
        gradcam_out = category_dir / f"{stem}_gradcam.png"
        self._save_heatmap_overlay(image_tensor, gradcam, gradcam_out)

        meta_out = category_dir / f"{stem}_interpretability.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": image_path,
                    "patch_shape": list(patch_shape),
                    "anomaly_map": anomaly_out.name,
                    "gradcam": gradcam_out.name,
                },
                f,
                indent=2,
            )

    def _run_shap_explainer(
        self,
        category: str,
        samples: List[Dict[str, object]],
    ) -> None:
        if not samples:
            return
        try:
            import shap
        except Exception as exc:
            logging.warning("SHAP not available (%s); skipping SHAP outputs.", exc)
            return

        memory_bank = self.memory_bank
        if memory_bank is None:
            logging.warning("Memory bank missing; skipping SHAP outputs.")
            return

        category_dir = self.output_dir / "interpretability" / category / "shap"
        category_dir.mkdir(parents=True, exist_ok=True)

        k = min(self.config.shap_max_patches, samples[0]["patch_scores"].numel())
        patch_scores = samples[0]["patch_scores"].flatten()
        topk = torch.topk(patch_scores, k=k).indices.cpu().numpy()
        feat_dim = samples[0]["embeddings"].shape[-1]

        def _flatten_sample(emb: Tensor) -> np.ndarray:
            selected = emb.reshape(-1, feat_dim)[topk]
            return selected.detach().cpu().numpy().reshape(1, -1)

        data = np.concatenate([_flatten_sample(s["embeddings"]) for s in samples], axis=0)
        background = data[: min(self.config.shap_background, data.shape[0])]
        memory_bank_cpu = memory_bank.detach().cpu()

        def model_fn(x: np.ndarray) -> np.ndarray:
            x_t = torch.from_numpy(x.astype(np.float32))
            n = x_t.shape[0]
            emb = x_t.view(n, k, feat_dim)
            flat = emb.reshape(-1, feat_dim)
            distances = torch.cdist(flat, memory_bank_cpu)
            min_distances, _ = torch.min(distances, dim=1)
            patch_scores = min_distances.view(n, k)
            k_top = max(1, int(0.01 * k))
            topk_vals = torch.topk(patch_scores, k=k_top, dim=1).values
            scores = topk_vals.mean(dim=1)
            return scores.detach().cpu().numpy()

        explainer = shap.KernelExplainer(model_fn, background)
        shap_values = explainer.shap_values(data, nsamples=100)
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[0]

        patch_h, patch_w = samples[0]["patch_shape"]
        for idx, sample in enumerate(samples):
            sv = shap_arr[idx].reshape(k, feat_dim)
            patch_importance = np.mean(np.abs(sv), axis=1)
            patch_map = np.zeros((patch_h * patch_w,), dtype=np.float32)
            patch_map[topk] = patch_importance
            patch_map = patch_map.reshape(patch_h, patch_w)
            patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-12)
            heatmap = (
                F.interpolate(
                    torch.from_numpy(patch_map).unsqueeze(0).unsqueeze(0),
                    size=(self.config.image_size, self.config.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )
            stem = Path(sample["image_path"]).stem
            out_path = category_dir / f"{stem}_shap.png"
            self._save_heatmap_overlay(sample["image_tensor"], heatmap, out_path)
            meta_path = category_dir / f"{stem}_shap.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": sample["image_path"],
                        "topk_patches": topk.tolist(),
                        "patch_importance": patch_importance.tolist(),
                        "shap_image": out_path.name,
                    },
                    f,
                    indent=2,
                )

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

        saliency_saved = 0
        shap_samples: List[Dict[str, object]] = []

        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating {category}")):
            if self.config.save_interpretability:
                images, labels, masks, paths = batch
            else:
                images, labels, masks = batch
            images = images.to(self.model.device)
            feats = self.model.extract(images)
            embeddings, patch_shape = self.model.aggregate(feats)
            anomaly_scores, anomaly_maps = self._compute_scores(embeddings, patch_shape)
            image_scores.extend(anomaly_scores.cpu().tolist())
            image_labels.extend(labels.cpu().tolist())

            if self.config.seg_masks:
                upsampled_maps = F.interpolate(
                    anomaly_maps,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for upsampled, mask in zip(upsampled_maps, masks):
                    pixel_scores.extend(upsampled.flatten().cpu().tolist())
                    pixel_labels.extend(mask.flatten().long().cpu().tolist())
            else:
                upsampled_maps = F.interpolate(
                    anomaly_maps,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if self.config.save_interpretability and saliency_saved < self.config.saliency_max_images:
                for idx in range(images.shape[0]):
                    if saliency_saved >= self.config.saliency_max_images:
                        break
                    self._save_interpretability_outputs(
                        category=category,
                        image_tensor=images[idx].detach().cpu(),
                        image_path=paths[idx],
                        anomaly_map=upsampled_maps[idx].detach().cpu(),
                        patch_shape=patch_shape,
                    )
                    saliency_saved += 1

            if self.config.save_interpretability and len(shap_samples) < self.config.shap_max_images:
                remaining = self.config.shap_max_images - len(shap_samples)
                for idx in range(min(remaining, images.shape[0])):
                    shap_samples.append(
                        {
                            "embeddings": embeddings[idx].detach().cpu(),
                            "patch_scores": anomaly_maps[idx].detach().cpu(),
                            "patch_shape": patch_shape,
                            "image_path": paths[idx],
                            "image_tensor": images[idx].detach().cpu(),
                        }
                    )

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
            "pixel_roc_auc": self._safe_auc(pixel_labels, pixel_scores) if self.config.seg_masks else float("nan"),
            "image_ap": self._safe_average_precision(image_labels, image_scores),
            "pixel_ap": self._safe_average_precision(image_labels, image_scores) if self.config.seg_masks else float("nan"),
        }

        del image_scores, image_labels, pixel_scores, pixel_labels, loader
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logging.info("Memory cleaned after %s evaluation", category)

        if self.config.save_interpretability and self.config.shap_max_images > 0:
            self._run_shap_explainer(category, shap_samples)

        return metrics_dict, self._count_test_images(category)

    def _count_test_images(self, category: str) -> int:
        ds = AutoVIAnomalyDataset(
            dataset_root=self.config.dataset_root,
            category=category,
            split="test",
            image_transform=self.eval_image_transform,
            mask_transform=self.mask_transform,
            seg_masks=self.config.seg_masks,
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
        if self.config.anomaly_score_clip is not None:
            patch_scores = torch.clamp(patch_scores, max=float(self.config.anomaly_score_clip))
        anomaly_maps = patch_scores.unsqueeze(1)
        k = max(1, int(0.01 * patch_scores.shape[-2] * patch_scores.shape[-1]))
        topk_vals = torch.topk(patch_scores.flatten(1), k=k, dim=1).values
        image_scores = topk_vals.mean(dim=1)
        if self.config.anomaly_score_clip is not None:
            image_scores = torch.clamp(image_scores, max=float(self.config.anomaly_score_clip))
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
    parser.add_argument("--interpretability", action="store_true")
    parser.add_argument("--saliency-max-images", type=int, default=10)
    parser.add_argument("--shap-max-images", type=int, default=0)
    parser.add_argument("--shap-background", type=int, default=20)
    parser.add_argument("--shap-max-patches", type=int, default=64)
    parser.add_argument("--corruption-prob", type=float, default=0.0)
    parser.add_argument("--corruption-strength", type=float, default=0.2)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.0)
    parser.add_argument("--robust-norm-max", type=float, default=None)
    parser.add_argument("--robust-cosine-min", type=float, default=None)
    parser.add_argument("--anomaly-score-clip", type=float, default=None)
    
    # NEW: Fairness arguments
    parser.add_argument(
        "--fairness-coreset-mode",
        type=str,
        default="none",
        choices=["none", "proportional", "equal"],
        help="Fairness-aware coreset sampling mode",
    )

    parser.add_argument("--seg-masks", action="store_true", default=True)
    parser.add_argument("--no-seg-masks", dest="seg_masks", action="store_false")
    
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
        save_interpretability=args.interpretability,
        saliency_max_images=args.saliency_max_images,
        shap_max_images=args.shap_max_images,
        shap_background=args.shap_background,
        shap_max_patches=args.shap_max_patches,
        seg_masks=args.seg_masks,
        corruption_prob=args.corruption_prob,
        corruption_strength=args.corruption_strength,
        gaussian_noise_std=args.gaussian_noise_std,
        robust_norm_max=args.robust_norm_max,
        robust_cosine_min=args.robust_cosine_min,
        anomaly_score_clip=args.anomaly_score_clip,
    )
    
    trainer = PatchCoreTrainer(config)
    summary: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    
    for category in categories:
        # Use fairness-aware coreset if specified
        checkpoint_path, _, _ = trainer.fit(
            category, 
            fairness_mode=args.fairness_coreset_mode
        )
        
        metrics_dict = None
        if not args.skip_eval:
            metrics_dict, _ = trainer.evaluate(category)
            logging.info("%s metrics: %s", category, metrics_dict)
        
        summary[category] = {"checkpoint": str(checkpoint_path), "metrics": metrics_dict}
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
