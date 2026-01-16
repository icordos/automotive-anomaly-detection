"""Fairness utilities for PatchCore anomaly detection.

Provides tools for:
- Subgroup detection from dataset structure
- Fairness-aware coreset sampling
- Fairness evaluation and reporting
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn import metrics


@dataclass
class SubgroupMetrics:
    """Metrics for a single subgroup."""
    subgroup_name: str
    n_samples: int
    n_anomalies: int
    image_auroc: float
    image_ap: float
    pixel_auroc: float
    pixel_ap: float
    tpr_at_fpr_01: float  # TPR @ FPR=0.1
    fpr_at_tpr_95: float  # FPR @ TPR=0.95


@dataclass
class FairnessReport:
    """Complete fairness evaluation report."""
    overall_metrics: Dict[str, float]
    subgroup_metrics: List[SubgroupMetrics]
    disparity_metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "overall": self.overall_metrics,
            "subgroups": [
                {
                    "name": sg.subgroup_name,
                    "n_samples": sg.n_samples,
                    "n_anomalies": sg.n_anomalies,
                    "image_auroc": sg.image_auroc,
                    "image_ap": sg.image_ap,
                    "pixel_auroc": sg.pixel_auroc,
                    "pixel_ap": sg.pixel_ap,
                    "tpr_at_fpr_01": sg.tpr_at_fpr_01,
                    "fpr_at_tpr_95": sg.fpr_at_tpr_95,
                }
                for sg in self.subgroup_metrics
            ],
            "disparities": self.disparity_metrics,
        }


class SubgroupDetector:
    """Automatically detect subgroups from dataset structure."""

    @staticmethod
    def detect_from_paths(image_paths: List[Path], category: str) -> Dict[str, List[int]]:
        """Detect subgroups based on folder structure or filename patterns.

        Args:
            image_paths: List of image file paths
            category: Category name

        Returns:
            dict: {subgroup_name: [image_indices]}
        """
        subgroups = defaultdict(list)

        for idx, path in enumerate(image_paths):
            # Strategy 1: Use defect type as subgroup (test/good vs test/scratch etc.)
            path_str = str(path)
            
            if "test" in path_str:
                parts = path.parts
                try:
                    test_idx = parts.index("test")
                    if test_idx + 1 < len(parts):
                        defect_type = parts[test_idx + 1]
                        subgroup_name = f"{category}_{defect_type}"
                    else:
                        subgroup_name = f"{category}_unknown"
                except ValueError:
                    subgroup_name = f"{category}_unknown"
            elif "train" in path_str:
                subgroup_name = f"{category}_train_good"
            else:
                subgroup_name = f"{category}_other"

            subgroups[subgroup_name].append(idx)

        # Log subgroup distribution
        logging.info("[SubgroupDetector] Detected %d subgroups for %s:", len(subgroups), category)
        for sg_name, indices in sorted(subgroups.items()):
            logging.info("  - %s: %d images", sg_name, len(indices))

        return dict(subgroups)

    @staticmethod
    def detect_from_metadata(
        image_paths: List[Path],
        metadata: Optional[Dict[str, str]] = None,
        category: str = "default",
    ) -> Dict[str, List[int]]:
        """Detect subgroups from external metadata (e.g., lighting conditions, production lines).

        Args:
            image_paths: List of image file paths
            metadata: Optional dict mapping filename -> subgroup_name
            category: Category name (fallback)

        Returns:
            dict: {subgroup_name: [image_indices]}
        """
        if metadata is None:
            return SubgroupDetector.detect_from_paths(image_paths, category)

        subgroups = defaultdict(list)
        for idx, path in enumerate(image_paths):
            key = str(path.name)
            subgroup_name = metadata.get(key, "unknown")
            subgroups[subgroup_name].append(idx)

        logging.info("[SubgroupDetector] Detected %d subgroups from metadata:", len(subgroups))
        for sg_name, indices in sorted(subgroups.items()):
            logging.info("  - %s: %d images", sg_name, len(indices))

        return dict(subgroups)


class FairnessEvaluator:
    """Evaluate fairness across subgroups."""

    @staticmethod
    def _safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
        """Compute AUROC with safety check for single-class case."""
        unique = set(labels)
        if len(unique) < 2:
            return float("nan")
        try:
            return float(metrics.roc_auc_score(labels, scores))
        except Exception as e:
            logging.warning("AUROC computation failed: %s", e)
            return float("nan")

    @staticmethod
    def _safe_ap(labels: Sequence[int], scores: Sequence[float]) -> float:
        """Compute Average Precision with safety check."""
        unique = set(labels)
        if len(unique) < 2:
            return float("nan")
        try:
            return float(metrics.average_precision_score(labels, scores))
        except Exception as e:
            logging.warning("AP computation failed: %s", e)
            return float("nan")

    @staticmethod
    def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.1) -> float:
        """Compute TPR at a given FPR threshold."""
        try:
            fpr, tpr, _ = metrics.roc_curve(labels, scores)
            idx = np.argmin(np.abs(fpr - target_fpr))
            return float(tpr[idx])
        except Exception as e:
            logging.warning("TPR@FPR computation failed: %s", e)
            return float("nan")

    @staticmethod
    def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
        """Compute FPR at a given TPR threshold."""
        try:
            fpr, tpr, _ = metrics.roc_curve(labels, scores)
            idx = np.argmin(np.abs(tpr - target_tpr))
            return float(fpr[idx])
        except Exception as e:
            logging.warning("FPR@TPR computation failed: %s", e)
            return float("nan")

    @staticmethod
    def evaluate_subgroup(
        subgroup_name: str,
        image_labels: List[int],
        image_scores: List[float],
        pixel_labels: List[int],
        pixel_scores: List[float],
    ) -> SubgroupMetrics:
        """Compute metrics for a single subgroup.

        Args:
            subgroup_name: Name of the subgroup
            image_labels: Image-level ground truth labels
            image_scores: Image-level anomaly scores
            pixel_labels: Pixel-level ground truth labels
            pixel_scores: Pixel-level anomaly scores

        Returns:
            SubgroupMetrics object
        """
        img_labels_arr = np.array(image_labels)
        img_scores_arr = np.array(image_scores)

        return SubgroupMetrics(
            subgroup_name=subgroup_name,
            n_samples=len(image_labels),
            n_anomalies=int(np.sum(img_labels_arr)),
            image_auroc=FairnessEvaluator._safe_auroc(image_labels, image_scores),
            image_ap=FairnessEvaluator._safe_ap(image_labels, image_scores),
            pixel_auroc=FairnessEvaluator._safe_auroc(pixel_labels, pixel_scores),
            pixel_ap=FairnessEvaluator._safe_ap(pixel_labels, pixel_scores),
            tpr_at_fpr_01=FairnessEvaluator._tpr_at_fpr(img_labels_arr, img_scores_arr, 0.1)
                if len(set(image_labels)) > 1 else float("nan"),
            fpr_at_tpr_95=FairnessEvaluator._fpr_at_tpr(img_labels_arr, img_scores_arr, 0.95)
                if len(set(image_labels)) > 1 else float("nan"),
        )

    @staticmethod
    def compute_disparities(subgroup_metrics: List[SubgroupMetrics]) -> Dict[str, float]:
        """Compute disparity metrics across subgroups.

        Metrics include:
        - AUROC range (max - min)
        - AUROC standard deviation
        - AUROC coefficient of variation
        - TPR disparity (equal opportunity violation)
        - FPR disparity

        Args:
            subgroup_metrics: List of SubgroupMetrics for all subgroups

        Returns:
            Dict of disparity metrics
        """
        valid_aurocs = [sg.image_auroc for sg in subgroup_metrics if not np.isnan(sg.image_auroc)]
        valid_tprs = [sg.tpr_at_fpr_01 for sg in subgroup_metrics if not np.isnan(sg.tpr_at_fpr_01)]
        valid_fprs = [sg.fpr_at_tpr_95 for sg in subgroup_metrics if not np.isnan(sg.fpr_at_tpr_95)]

        disparities = {}

        if len(valid_aurocs) > 1:
            disparities["auroc_range"] = float(np.max(valid_aurocs) - np.min(valid_aurocs))
            disparities["auroc_std"] = float(np.std(valid_aurocs))
            disparities["auroc_mean"] = float(np.mean(valid_aurocs))
            mean_auroc = np.mean(valid_aurocs)
            if mean_auroc > 0:
                disparities["auroc_cv"] = float(np.std(valid_aurocs) / mean_auroc)

        if len(valid_tprs) > 1:
            disparities["delta_tpr"] = float(np.max(valid_tprs) - np.min(valid_tprs))
            disparities["tpr_std"] = float(np.std(valid_tprs))
            disparities["tpr_mean"] = float(np.mean(valid_tprs))

        if len(valid_fprs) > 1:
            disparities["delta_fpr"] = float(np.max(valid_fprs) - np.min(valid_fprs))
            disparities["fpr_std"] = float(np.std(valid_fprs))

        # Equal Opportunity: max difference in TPR among anomaly classes
        anomaly_subgroups = [sg for sg in subgroup_metrics if sg.n_anomalies > 0]
        if len(anomaly_subgroups) > 1:
            anomaly_tprs = [sg.tpr_at_fpr_01 for sg in anomaly_subgroups if not np.isnan(sg.tpr_at_fpr_01)]
            if len(anomaly_tprs) > 1:
                disparities["equal_opportunity_violation"] = float(np.max(anomaly_tprs) - np.min(anomaly_tprs))

        # Demographic Parity: variance in positive prediction rates across subgroups
        # (Using FPR as proxy for prediction rate in normal samples)
        normal_subgroups = [sg for sg in subgroup_metrics if sg.n_samples > sg.n_anomalies]
        if len(normal_subgroups) > 1:
            normal_fprs = [sg.fpr_at_tpr_95 for sg in normal_subgroups if not np.isnan(sg.fpr_at_tpr_95)]
            if len(normal_fprs) > 1:
                disparities["demographic_parity_violation"] = float(np.std(normal_fprs))

        return disparities

    @staticmethod
    def generate_report(
        overall_image_labels: List[int],
        overall_image_scores: List[float],
        overall_pixel_labels: List[int],
        overall_pixel_scores: List[float],
        subgroup_assignments: Dict[str, List[int]],
        all_image_labels: List[int],
        all_image_scores: List[float],
        all_pixel_labels_by_image: List[List[int]],
        all_pixel_scores_by_image: List[List[float]],
    ) -> FairnessReport:
        """Generate complete fairness report.

        Args:
            overall_image_labels: All image labels (flattened)
            overall_image_scores: All image scores (flattened)
            overall_pixel_labels: All pixel labels (flattened)
            overall_pixel_scores: All pixel scores (flattened)
            subgroup_assignments: Dict mapping subgroup_name -> [image_indices]
            all_image_labels: Per-image labels (indexed)
            all_image_scores: Per-image scores (indexed)
            all_pixel_labels_by_image: Pixel labels grouped by image
            all_pixel_scores_by_image: Pixel scores grouped by image

        Returns:
            FairnessReport object
        """
        # Overall metrics
        overall = {
            "image_auroc": FairnessEvaluator._safe_auroc(overall_image_labels, overall_image_scores),
            "image_ap": FairnessEvaluator._safe_ap(overall_image_labels, overall_image_scores),
            "pixel_auroc": FairnessEvaluator._safe_auroc(overall_pixel_labels, overall_pixel_scores),
            "pixel_ap": FairnessEvaluator._safe_ap(overall_pixel_labels, overall_pixel_scores),
        }

        # Per-subgroup metrics
        subgroup_metrics_list = []
        for subgroup_name, indices in subgroup_assignments.items():
            if not indices:
                continue

            sg_img_labels = [all_image_labels[i] for i in indices if i < len(all_image_labels)]
            sg_img_scores = [all_image_scores[i] for i in indices if i < len(all_image_scores)]
            
            if not sg_img_labels:
                logging.warning("Subgroup %s has no valid samples, skipping", subgroup_name)
                continue
            
            sg_pix_labels = []
            sg_pix_scores = []
            for i in indices:
                if i < len(all_pixel_labels_by_image):
                    sg_pix_labels.extend(all_pixel_labels_by_image[i])
                    sg_pix_scores.extend(all_pixel_scores_by_image[i])

            sg_metrics = FairnessEvaluator.evaluate_subgroup(
                subgroup_name, sg_img_labels, sg_img_scores, sg_pix_labels, sg_pix_scores
            )
            subgroup_metrics_list.append(sg_metrics)

        # Disparity metrics
        disparities = FairnessEvaluator.compute_disparities(subgroup_metrics_list)

        return FairnessReport(
            overall_metrics=overall,
            subgroup_metrics=subgroup_metrics_list,
            disparity_metrics=disparities,
        )


class FairCoreset:
    """Fairness-aware coreset sampling."""

    @staticmethod
    def balanced_sampling(
        features: torch.Tensor,
        subgroup_labels: np.ndarray,
        target_size: int,
        method: str = "proportional",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample coreset with balanced subgroup representation.

        Args:
            features: (N, D) feature tensor
            subgroup_labels: (N,) array of subgroup IDs (integers)
            target_size: target coreset size
            method: 'proportional' (keep original ratios) or 'equal' (equal per group)
            seed: Random seed for reproducibility

        Returns:
            Tensor of selected indices (dtype=long)
        """
        if seed is not None:
            np.random.seed(seed)
        
        unique_groups = np.unique(subgroup_labels)
        n_groups = len(unique_groups)

        if n_groups == 0:
            raise ValueError("No subgroups found in subgroup_labels")

        logging.info(
            "[FairCoreset] Sampling %d features from %d subgroups using method='%s'",
            target_size,
            n_groups,
            method,
        )

        if method == "equal":
            samples_per_group = max(1, target_size // n_groups)
            group_targets = {g: samples_per_group for g in unique_groups}
        elif method == "proportional":
            group_counts = {g: int(np.sum(subgroup_labels == g)) for g in unique_groups}
            total = len(subgroup_labels)
            group_targets = {
                g: max(1, int(target_size * count / total)) 
                for g, count in group_counts.items()
            }
        else:
            raise ValueError(f"Unknown method: {method}. Use 'proportional' or 'equal'.")

        selected_indices = []

        for group_id in unique_groups:
            group_mask = subgroup_labels == group_id
            group_indices = np.where(group_mask)[0]

            n_select = min(group_targets[group_id], len(group_indices))

            # Random sampling within group
            selected = np.random.choice(group_indices, size=n_select, replace=False)
            selected_indices.extend(selected.tolist())

            logging.debug(
                "[FairCoreset] Subgroup %d: sampled %d / %d (target=%d)",
                group_id,
                n_select,
                len(group_indices),
                group_targets[group_id],
            )

        # If we're under target, sample more uniformly from remaining
        if len(selected_indices) < target_size:
            remaining = target_size - len(selected_indices)
            all_indices = np.arange(len(subgroup_labels))
            available = np.setdiff1d(all_indices, selected_indices)
            
            if len(available) > 0:
                extra = np.random.choice(
                    available, 
                    size=min(remaining, len(available)), 
                    replace=False
                )
                selected_indices.extend(extra.tolist())
                logging.debug(
                    "[FairCoreset] Added %d extra samples to reach target %d",
                    len(extra),
                    target_size,
                )

        logging.info(
            "[FairCoreset] Final selection: %d indices (target was %d)",
            len(selected_indices),
            target_size,
        )

        return torch.tensor(selected_indices, dtype=torch.long)


def save_fairness_report(report: FairnessReport, output_path: Path) -> None:
    """Save fairness report to JSON.

    Args:
        report: FairnessReport object
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logging.info("Saved fairness report to %s", output_path)


def log_fairness_summary(report: FairnessReport) -> None:
    """Log fairness summary to console.

    Args:
        report: FairnessReport object
    """
    logging.info("=" * 60)
    logging.info("FAIRNESS EVALUATION SUMMARY")
    logging.info("=" * 60)

    logging.info("Overall Metrics:")
    for k, v in report.overall_metrics.items():
        if not np.isnan(v):
            logging.info("  %s: %.4f", k, v)
        else:
            logging.info("  %s: N/A", k)

    logging.info("\nPer-Subgroup Metrics:")
    for sg in report.subgroup_metrics:
        logging.info("\n  %s:", sg.subgroup_name)
        logging.info("    Samples: %d (Anomalies: %d)", sg.n_samples, sg.n_anomalies)
        if not np.isnan(sg.image_auroc):
            logging.info("    Image AUROC: %.4f", sg.image_auroc)
        if not np.isnan(sg.tpr_at_fpr_01):
            logging.info("    TPR@FPR=0.1: %.4f", sg.tpr_at_fpr_01)
        if not np.isnan(sg.fpr_at_tpr_95):
            logging.info("    FPR@TPR=0.95: %.4f", sg.fpr_at_tpr_95)

    logging.info("\nDisparity Metrics:")
    for k, v in report.disparity_metrics.items():
        logging.info("  %s: %.4f", k, v)

    logging.info("=" * 60)


# Example usage / testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Test subgroup detection
    test_paths = [
        Path("data/raw/category/category/train/good/001.png"),
        Path("data/raw/category/category/train/good/002.png"),
        Path("data/raw/category/category/test/good/003.png"),
        Path("data/raw/category/category/test/scratch/004.png"),
        Path("data/raw/category/category/test/scratch/005.png"),
    ]

    subgroups = SubgroupDetector.detect_from_paths(test_paths, "category")
    print("\nDetected subgroups:")
    for name, indices in subgroups.items():
        print(f"  {name}: {indices}")

    # Test fairness-aware sampling
    features = torch.randn(100, 512)
    labels = np.array([0] * 60 + [1] * 30 + [2] * 10)  # Imbalanced subgroups

    print("\n--- Equal sampling ---")
    selected_equal = FairCoreset.balanced_sampling(features, labels, target_size=30, method="equal", seed=42)
    print(f"Selected {len(selected_equal)} indices")
    for sg in [0, 1, 2]:
        count = np.sum(labels[selected_equal.numpy()] == sg)
        print(f"  Subgroup {sg}: {count} samples")

    print("\n--- Proportional sampling ---")
    selected_prop = FairCoreset.balanced_sampling(features, labels, target_size=30, method="proportional", seed=42)
    print(f"Selected {len(selected_prop)} indices")
    for sg in [0, 1, 2]:
        count = np.sum(labels[selected_prop.numpy()] == sg)
        print(f"  Subgroup {sg}: {count} samples")
