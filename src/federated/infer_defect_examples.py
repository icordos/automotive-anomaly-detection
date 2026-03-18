#!/usr/bin/env python
"""Run inference on per-category defect_example.png files using saved PatchCore banks."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from patchcore_training import PatchCoreTrainer, PatchCoreTrainingConfig, discover_categories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference only on defect_example.png files using existing PatchCore checkpoints."
    )
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    p.add_argument("--categories", nargs="*", help="Categories to process; defaults to all discovered")
    p.add_argument(
        "--checkpoint-roots",
        nargs="+",
        type=Path,
        default=[Path("artifacts")],
        help="Directories searched recursively for <category>_patchcore.pt checkpoints",
    )
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/defect-example-inference"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--distance-chunk-size", type=int, default=8192)
    p.add_argument("--interpretability", action="store_true", default=True)
    p.add_argument("--saliency-max-images", type=int, default=1)
    p.add_argument("--shap-max-images", type=int, default=0)
    p.add_argument("--shap-background", type=int, default=20)
    p.add_argument("--shap-max-patches", type=int, default=64)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def find_checkpoint(category: str, roots: List[Path]) -> Path:
    name = f"{category}_patchcore.pt"
    matches: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        matches.extend(sorted(root.rglob(name)))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint named {name} found under: {', '.join(str(root) for root in roots)}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple checkpoints found for {category}: {', '.join(str(m) for m in matches)}. "
            "Pass a narrower --checkpoint-roots path."
        )
    return matches[0]


def resolve_example_path(dataset_root: Path, category: str) -> Path:
    p1 = dataset_root / category / category / "defect_example.png"
    if p1.exists():
        return p1
    p2 = dataset_root / category / "defect_example.png"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing defect_example.png for {category} under {dataset_root}")


def load_checkpoint(checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "memory_bank" not in checkpoint:
        raise ValueError(f"Checkpoint missing memory_bank: {checkpoint_path}")
    return checkpoint


def build_trainer(
    checkpoint: Dict[str, object],
    dataset_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> PatchCoreTrainer:
    image_size = int(checkpoint.get("image_size", 512))
    layers = tuple(checkpoint.get("layers", ("layer2", "layer3")))
    config = PatchCoreTrainingConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        image_size=image_size,
        batch_size=1,
        num_workers=args.num_workers,
        device=args.device,
        feature_layers=layers,
        distance_chunk_size=args.distance_chunk_size,
        save_interpretability=args.interpretability,
        saliency_max_images=args.saliency_max_images,
        shap_max_images=args.shap_max_images,
        shap_background=args.shap_background,
        shap_max_patches=args.shap_max_patches,
        seg_masks=False,
    )
    trainer = PatchCoreTrainer(config)
    trainer.load_memory_bank(checkpoint["memory_bank"])
    return trainer


def run_single_image(
    trainer: PatchCoreTrainer,
    category: str,
    image_path: Path,
) -> Dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = trainer.eval_image_transform(image)
    batch = image_tensor.unsqueeze(0).to(trainer.model.device)

    feats = trainer.model.extract(batch)
    embeddings, patch_shape = trainer.model.aggregate(feats)
    image_scores, anomaly_maps = trainer._compute_scores(embeddings, patch_shape)
    upsampled_map = F.interpolate(
        anomaly_maps,
        size=(trainer.config.image_size, trainer.config.image_size),
        mode="bilinear",
        align_corners=False,
    )[0].detach().cpu()

    if trainer.config.save_interpretability and trainer.config.saliency_max_images > 0:
        trainer._save_interpretability_outputs(
            category=category,
            image_tensor=image_tensor,
            image_path=str(image_path),
            anomaly_map=upsampled_map,
            patch_scores=anomaly_maps[0].detach().cpu(),
            patch_shape=patch_shape,
        )

    if trainer.config.save_interpretability and trainer.config.shap_max_images > 0:
        trainer._run_shap_explainer(
            category,
            [
                {
                    "embeddings": embeddings[0].detach().cpu(),
                    "patch_scores": anomaly_maps[0].detach().cpu(),
                    "patch_shape": patch_shape,
                    "image_path": str(image_path),
                    "image_tensor": image_tensor,
                }
            ],
        )

    return {
        "image": str(image_path),
        "checkpoint": None,
        "image_score": float(image_scores[0].detach().cpu().item()),
        "patch_shape": list(patch_shape),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    categories = args.categories or discover_categories(args.dataset_root)
    if not categories:
        raise ValueError(f"No categories found in {args.dataset_root}")

    summary: Dict[str, Dict[str, object]] = {}
    for category in categories:
        checkpoint_path = find_checkpoint(category, args.checkpoint_roots)
        example_path = resolve_example_path(args.dataset_root, category)
        category_output_dir = args.output_dir

        logging.info("Category %s", category)
        logging.info("Using checkpoint: %s", checkpoint_path)
        logging.info("Using example image: %s", example_path)

        checkpoint = load_checkpoint(checkpoint_path)
        trainer = build_trainer(checkpoint, args.dataset_root, category_output_dir, args)
        result = run_single_image(trainer, category, example_path)
        result["checkpoint"] = str(checkpoint_path)
        summary[category] = result

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
