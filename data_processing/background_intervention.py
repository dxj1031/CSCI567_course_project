#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from _common import (
    build_master_table,
    build_variant_root,
    copy_processed_metadata,
    load_processed_tables,
    resolve_source_images_dir,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a SAM-based background-suppressed CCT20 dataset variant."
    )
    parser.add_argument("--source-root", required=True, help="Original dataset root containing images/ and processed/.")
    parser.add_argument("--output-root", required=True, help="Directory where dataset variants should be stored.")
    parser.add_argument("--variant-name", default="dataset_sam_bg", help="Name of the output dataset variant.")
    parser.add_argument(
        "--sam-checkpoint",
        required=True,
        help="Path to a Segment Anything checkpoint file such as sam_vit_h_4b8939.pth.",
    )
    parser.add_argument(
        "--model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM backbone type matching the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device to run SAM on. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed used for reproducibility.")
    parser.add_argument("--blur-radius", type=float, default=8.0, help="Gaussian blur radius applied to the background.")
    parser.add_argument(
        "--mask-feather",
        type=float,
        default=4.0,
        help="Blur radius applied to the selected foreground mask for smoother edges.",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=24,
        help="SAM automatic mask generator points_per_side parameter.",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.88,
        help="SAM automatic mask generator pred_iou_thresh parameter.",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.95,
        help="SAM automatic mask generator stability_score_thresh parameter.",
    )
    parser.add_argument(
        "--box-nms-thresh",
        type=float,
        default=0.7,
        help="SAM automatic mask generator box_nms_thresh parameter.",
    )
    parser.add_argument(
        "--central-box-scale",
        type=float,
        default=0.55,
        help="Relative width/height of the central box used to pick the foreground mask.",
    )
    parser.add_argument(
        "--min-mask-area-fraction",
        type=float,
        default=0.003,
        help="Ignore masks smaller than this fraction of the full image area.",
    )
    return parser.parse_args()


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_mask_generator(
    checkpoint_path: Path,
    model_type: str,
    device_name: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    box_nms_thresh: float,
):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment_anything is required for the SAM background intervention. "
            "Install it in the CARC environment before running this script."
        ) from exc

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device_name)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
    )


def central_box_bounds(width: int, height: int, central_box_scale: float) -> tuple[int, int, int, int]:
    box_width = max(1, int(round(width * central_box_scale)))
    box_height = max(1, int(round(height * central_box_scale)))
    left = max(0, (width - box_width) // 2)
    top = max(0, (height - box_height) // 2)
    right = min(width, left + box_width)
    bottom = min(height, top + box_height)
    return left, top, right, bottom


def build_fallback_mask(width: int, height: int, central_box_scale: float) -> np.ndarray:
    left, top, right, bottom = central_box_bounds(width, height, central_box_scale)
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def score_mask(
    candidate: dict[str, Any],
    width: int,
    height: int,
    central_box_scale: float,
    min_mask_area_fraction: float,
) -> dict[str, Any] | None:
    segmentation = candidate["segmentation"].astype(bool)
    image_area = float(width * height)
    area = float(candidate.get("area", float(segmentation.sum())))
    area_fraction = area / image_area
    if area_fraction < min_mask_area_fraction:
        return None

    left, top, right, bottom = central_box_bounds(width, height, central_box_scale)
    central_crop = segmentation[top:bottom, left:right]
    central_overlap = float(central_crop.mean()) if central_crop.size else 0.0
    center_pixel = bool(segmentation[height // 2, width // 2])
    large_mask_penalty = max(area_fraction - 0.65, 0.0)
    ideal_area_score = 1.0 - min(abs(area_fraction - 0.18) / 0.18, 1.0)
    pred_iou = float(candidate.get("predicted_iou", 0.0))
    stability_score = float(candidate.get("stability_score", 0.0))

    score = (
        3.0 * central_overlap
        + 1.2 * float(center_pixel)
        + 0.8 * ideal_area_score
        + 0.5 * pred_iou
        + 0.5 * stability_score
        - 1.5 * large_mask_penalty
    )

    return {
        "score": score,
        "mask": segmentation,
        "mask_area_fraction": area_fraction,
        "central_overlap": central_overlap,
        "predicted_iou": pred_iou,
        "stability_score": stability_score,
    }


def select_foreground_mask(
    mask_candidates: list[dict[str, Any]],
    width: int,
    height: int,
    central_box_scale: float,
    min_mask_area_fraction: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for candidate in mask_candidates:
        scored = score_mask(candidate, width, height, central_box_scale, min_mask_area_fraction)
        if scored is None:
            continue
        if best is None or scored["score"] > best["score"]:
            best = scored

    if best is not None:
        best["selection_source"] = "sam"
        return best

    fallback_mask = build_fallback_mask(width, height, central_box_scale)
    return {
        "score": 0.0,
        "mask": fallback_mask,
        "mask_area_fraction": float(fallback_mask.mean()),
        "central_overlap": 1.0,
        "predicted_iou": None,
        "stability_score": None,
        "selection_source": "fallback_central_box",
    }


def apply_background_suppression(
    image: Image.Image,
    foreground_mask: np.ndarray,
    blur_radius: float,
    mask_feather: float,
) -> Image.Image:
    image = image.convert("RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_image = Image.fromarray((foreground_mask.astype(np.uint8) * 255), mode="L")
    if mask_feather > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_feather))
    return Image.composite(image, blurred, mask_image)


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint_path = Path(args.sam_checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    configure_determinism(args.seed)
    device_name = resolve_device(args.device)

    tables = load_processed_tables(source_root)
    master = build_master_table(tables)
    source_images_dir = resolve_source_images_dir(source_root, master["file_name"].head(32).tolist())
    variant_root = build_variant_root(output_root, args.variant_name)
    copy_processed_metadata(source_root, variant_root)
    output_images_dir = variant_root / "images"

    mask_generator = build_mask_generator(
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        device_name=device_name,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
    )

    selection_rows: list[dict[str, Any]] = []
    fallback_count = 0

    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        output_path = output_images_dir / row.file_name

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            rgb = np.asarray(image)
            height, width = rgb.shape[:2]
            mask_candidates = mask_generator.generate(rgb)
            selected = select_foreground_mask(
                mask_candidates=mask_candidates,
                width=width,
                height=height,
                central_box_scale=args.central_box_scale,
                min_mask_area_fraction=args.min_mask_area_fraction,
            )
            transformed = apply_background_suppression(
                image=image,
                foreground_mask=selected["mask"],
                blur_radius=args.blur_radius,
                mask_feather=args.mask_feather,
            )
            save_image(transformed, output_path)

        if selected["selection_source"] != "sam":
            fallback_count += 1

        selection_rows.append(
            {
                "file_name": row.file_name,
                "split": getattr(row, "split", None),
                "day_night": getattr(row, "day_night", None),
                "selection_source": selected["selection_source"],
                "mask_area_fraction": selected["mask_area_fraction"],
                "central_overlap": selected["central_overlap"],
                "predicted_iou": selected["predicted_iou"],
                "stability_score": selected["stability_score"],
            }
        )

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(variant_root / "metadata" / "sam_mask_selection.csv", index=False)

    metadata = {
        "variant_name": args.variant_name,
        "source_root": str(source_root),
        "resolved_source_images_dir": str(source_images_dir),
        "variant_root": str(variant_root),
        "num_images_processed": int(len(master)),
        "sam": {
            "checkpoint": str(checkpoint_path),
            "model_type": args.model_type,
            "device": device_name,
            "seed": args.seed,
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "box_nms_thresh": args.box_nms_thresh,
        },
        "intervention": {
            "type": "sam_background_suppression",
            "selection_strategy": "automatic_mask_generation_with_largest_central_object_heuristic",
            "blur_radius": args.blur_radius,
            "mask_feather": args.mask_feather,
            "central_box_scale": args.central_box_scale,
            "min_mask_area_fraction": args.min_mask_area_fraction,
            "fallback_central_box_count": fallback_count,
        },
    }
    save_json(variant_root / "metadata" / "background_intervention.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
