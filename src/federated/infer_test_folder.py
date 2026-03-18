#!/usr/bin/env python3
"""Run inference on category test folders using saved PatchCore checkpoints."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from patchcore_training import PatchCoreTrainer, PatchCoreTrainingConfig, discover_categories


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference on all images in each category test folder using existing PatchCore checkpoints."
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
    p.add_argument("--output-dir", type=Path, default=Path("inference"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--distance-chunk-size", type=int, default=8192)
    p.add_argument("--patch-quality-top-percent", type=float, default=1.0)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def find_checkpoint(category: str, roots: List[Path]) -> Path:
    name = f"{category}_patchcore.pt"
    matches: List[Path] = []
    for root in roots:
        if root.exists():
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


def resolve_category_dir(dataset_root: Path, category: str) -> Path:
    p1 = dataset_root / category / category
    if p1.exists():
        return p1
    p2 = dataset_root / category
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing category directory for {category} under {dataset_root}")


def iter_test_images(category_dir: Path) -> Iterable[Path]:
    test_dir = category_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    for path in sorted(test_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


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
    config = PatchCoreTrainingConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        image_size=int(checkpoint.get("image_size", 512)),
        batch_size=1,
        num_workers=args.num_workers,
        device=args.device,
        feature_layers=tuple(checkpoint.get("layers", ("layer2", "layer3"))),
        distance_chunk_size=args.distance_chunk_size,
        save_interpretability=False,
        seg_masks=False,
        patch_quality_top_percent=args.patch_quality_top_percent,
    )
    trainer = PatchCoreTrainer(config)
    trainer.load_memory_bank(checkpoint["memory_bank"])
    return trainer


def save_side_by_side(
    original_path: Path,
    anomaly_path: Path,
    gradcam_path: Path,
    patch_quality_path: Path,
    out_path: Path,
) -> None:
    panels = [
        ("Original", Image.open(original_path).convert("RGB")),
        ("Anomaly", Image.open(anomaly_path).convert("RGB")),
        ("GradCAM", Image.open(gradcam_path).convert("RGB")),
        ("PatchQuality", Image.open(patch_quality_path).convert("RGB")),
    ]

    label_height = 28
    target_h = max(img.height for _, img in panels)
    resized = []
    for label, img in panels:
        if img.height != target_h:
            width = int(round(img.width * (target_h / img.height)))
            img = img.resize((width, target_h), Image.Resampling.BILINEAR)
        resized.append((label, img))

    total_w = sum(img.width for _, img in resized)
    canvas = Image.new("RGB", (total_w, target_h + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, img in resized:
        canvas.paste(img, (x, label_height))
        draw.text((x + 8, 6), label, fill=(0, 0, 0))
        x += img.width

    canvas.save(out_path)


def run_single_image(
    trainer: PatchCoreTrainer,
    category: str,
    source_path: Path,
    destination_dir: Path,
) -> Dict[str, object]:
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_original = destination_dir / source_path.name
    shutil.copy2(source_path, copied_original)

    image = Image.open(source_path).convert("RGB")
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

    stem = source_path.stem
    anomaly_np = upsampled_map.squeeze(0).numpy()
    anomaly_np = (anomaly_np - anomaly_np.min()) / (anomaly_np.max() - anomaly_np.min() + 1e-12)
    anomaly_path = destination_dir / f"{stem}_anomaly.png"
    trainer._save_heatmap_overlay(image_tensor, anomaly_np, anomaly_path)

    gradcam = trainer._compute_gradcam(image_tensor)
    gradcam_path = destination_dir / f"{stem}_gradcam.png"
    trainer._save_heatmap_overlay(image_tensor, gradcam, gradcam_path)

    patch_quality_path = destination_dir / f"{stem}_patch_quality.png"
    patch_quality_overlay_only_path = destination_dir / f"{stem}_patch_quality_overlay_only.png"
    trainer._save_patch_quality_overlay(
        image_tensor,
        anomaly_maps[0].detach().cpu(),
        patch_quality_path,
        overlay_only_path=patch_quality_overlay_only_path,
    )

    side_by_side_path = destination_dir / f"{stem}_side_by_side.png"
    save_side_by_side(
        copied_original,
        anomaly_path,
        gradcam_path,
        patch_quality_path,
        side_by_side_path,
    )

    meta_path = destination_dir / f"{stem}_inference.json"
    payload = {
        "category": category,
        "image": str(source_path),
        "copied_original": copied_original.name,
        "image_score": float(image_scores[0].detach().cpu().item()),
        "patch_shape": list(patch_shape),
        "patch_quality_rule": f"top_{trainer.config.patch_quality_top_percent:g}_percent_patch_scores_red_rest_green",
        "anomaly": anomaly_path.name,
        "gradcam": gradcam_path.name,
        "patch_quality": patch_quality_path.name,
        "patch_quality_overlay_only": patch_quality_overlay_only_path.name,
        "side_by_side": side_by_side_path.name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    payload["metadata"] = meta_path.name
    return payload


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    categories = args.categories or discover_categories(args.dataset_root)
    if not categories:
        raise ValueError(f"No categories found in {args.dataset_root}")

    summary: Dict[str, List[Dict[str, object]]] = {}

    for category in categories:
        checkpoint_path = find_checkpoint(category, args.checkpoint_roots)
        category_dir = resolve_category_dir(args.dataset_root, category)
        trainer = build_trainer(load_checkpoint(checkpoint_path), args.dataset_root, args.output_dir, args)

        logging.info("Category %s using checkpoint %s", category, checkpoint_path)
        category_results: List[Dict[str, object]] = []
        for image_path in iter_test_images(category_dir):
            relative_path = image_path.relative_to(category_dir)
            destination_dir = args.output_dir / category / relative_path.parent
            logging.info("Infer %s", image_path)
            result = run_single_image(trainer, category, image_path, destination_dir)
            result["checkpoint"] = str(checkpoint_path)
            category_results.append(result)

        summary[category] = category_results
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"summary": str(summary_path), "categories": categories}, indent=2))


if __name__ == "__main__":
    main()
