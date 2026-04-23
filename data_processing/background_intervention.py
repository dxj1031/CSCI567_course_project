#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import tarfile
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


ANNOTATION_JSON_NAMES = [
    "train_annotations.json",
    "cis_val_annotations.json",
    "trans_val_annotations.json",
    "cis_test_annotations.json",
    "trans_test_annotations.json",
]


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
        "--central-box-scale",
        type=float,
        default=0.55,
        help="Relative width/height of the central fallback box used when no annotation bbox is available.",
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


def build_predictor(checkpoint_path: Path, model_type: str, device_name: str):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment_anything is required for the SAM background intervention. "
            "Install it in the CARC environment before running this script."
        ) from exc

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device_name)
    return SamPredictor(sam)


def central_box_bounds(width: int, height: int, central_box_scale: float) -> tuple[int, int, int, int]:
    box_width = max(1, int(round(width * central_box_scale)))
    box_height = max(1, int(round(height * central_box_scale)))
    left = max(0, (width - box_width) // 2)
    top = max(0, (height - box_height) // 2)
    right = min(width, left + box_width)
    bottom = min(height, top + box_height)
    return left, top, right, bottom


def xywh_to_xyxy(bbox: list[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = bbox
    x0 = max(0.0, min(float(width - 1), x))
    y0 = max(0.0, min(float(height - 1), y))
    x1 = max(x0 + 1.0, min(float(width), x + w))
    y1 = max(y0 + 1.0, min(float(height), y + h))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def union_bboxes(bboxes: list[list[float]], width: int, height: int) -> np.ndarray:
    xyxy_boxes = np.array([xywh_to_xyxy(bbox, width, height) for bbox in bboxes], dtype=np.float32)
    return np.array(
        [
            float(xyxy_boxes[:, 0].min()),
            float(xyxy_boxes[:, 1].min()),
            float(xyxy_boxes[:, 2].max()),
            float(xyxy_boxes[:, 3].max()),
        ],
        dtype=np.float32,
    )


def load_annotation_payloads(source_root: Path) -> list[dict[str, Any]]:
    annotations_dir = source_root / "annotations"
    archive_path = annotations_dir / "eccv_18_annotations.tar.gz"
    payloads: list[dict[str, Any]] = []

    if archive_path.exists():
        with tarfile.open(archive_path, "r:gz") as archive:
            member_names = {member.name for member in archive.getmembers()}
            for json_name in ANNOTATION_JSON_NAMES:
                member_name = f"eccv_18_annotation_files/{json_name}"
                if member_name not in member_names:
                    continue
                extracted = archive.extractfile(member_name)
                if extracted is None:
                    continue
                payloads.append(json.load(extracted))
        if payloads:
            return payloads

    for json_name in ANNOTATION_JSON_NAMES:
        direct_path = annotations_dir / json_name
        if direct_path.exists():
            payloads.append(json.loads(direct_path.read_text(encoding="utf-8")))

    return payloads


def build_bbox_index(source_root: Path) -> dict[str, list[list[float]]]:
    payloads = load_annotation_payloads(source_root)
    bbox_index: dict[str, list[list[float]]] = {}

    for payload in payloads:
        for annotation in payload.get("annotations", []):
            image_id = str(annotation.get("image_id", ""))
            bbox = annotation.get("bbox")
            if not image_id or bbox is None:
                continue
            bbox_index.setdefault(image_id, []).append([float(value) for value in bbox])

    return bbox_index


def choose_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    min_mask_area_fraction: float,
) -> tuple[np.ndarray, float]:
    best_mask: np.ndarray | None = None
    best_score = float("-inf")
    image_area = float(width * height)

    for mask, score in zip(masks, scores):
        mask = mask.astype(bool)
        area_fraction = float(mask.sum()) / image_area
        if area_fraction < min_mask_area_fraction:
            continue

        too_large_penalty = max(area_fraction - 0.8, 0.0)
        adjusted_score = float(score) - 0.5 * too_large_penalty
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_mask = mask

    if best_mask is None:
        best_index = int(np.argmax(scores))
        best_mask = masks[best_index].astype(bool)
        best_score = float(scores[best_index])

    return best_mask, best_score


def build_prompt_points(width: int, height: int, box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box_xyxy.tolist()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    margin = 10.0
    negative_points = [
        [margin, margin],
        [width - margin, margin],
        [margin, height - margin],
        [width - margin, height - margin],
    ]
    point_coords = np.array([[cx, cy], *negative_points], dtype=np.float32)
    point_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    return point_coords, point_labels


def predict_mask_from_box(
    predictor,
    rgb: np.ndarray,
    box_xyxy: np.ndarray,
    min_mask_area_fraction: float,
) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    predictor.set_image(rgb)
    point_coords, point_labels = build_prompt_points(width, height, box_xyxy)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_xyxy,
        multimask_output=True,
    )
    return choose_best_mask(masks, scores, width, height, min_mask_area_fraction)


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

    bbox_index = build_bbox_index(source_root)
    predictor = build_predictor(
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        device_name=device_name,
    )

    selection_rows: list[dict[str, Any]] = []
    fallback_count = 0
    annotation_prompt_count = 0

    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        output_path = output_images_dir / row.file_name
        image_id = Path(row.file_name).stem

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            rgb = np.asarray(image)
            height, width = rgb.shape[:2]

            bboxes = bbox_index.get(image_id, [])
            if bboxes:
                box_xyxy = union_bboxes(bboxes, width, height)
                selection_source = "annotation_bbox_prompt"
                annotation_prompt_count += 1
            else:
                box_xyxy = np.array(central_box_bounds(width, height, args.central_box_scale), dtype=np.float32)
                selection_source = "fallback_central_box_prompt"
                fallback_count += 1

            foreground_mask, sam_score = predict_mask_from_box(
                predictor=predictor,
                rgb=rgb,
                box_xyxy=box_xyxy,
                min_mask_area_fraction=args.min_mask_area_fraction,
            )

            transformed = apply_background_suppression(
                image=image,
                foreground_mask=foreground_mask,
                blur_radius=args.blur_radius,
                mask_feather=args.mask_feather,
            )
            save_image(transformed, output_path)

        selection_rows.append(
            {
                "file_name": row.file_name,
                "split": getattr(row, "split", None),
                "day_night": getattr(row, "day_night", None),
                "selection_source": selection_source,
                "mask_area_fraction": float(foreground_mask.mean()),
                "sam_score": sam_score,
                "prompt_box_x0": float(box_xyxy[0]),
                "prompt_box_y0": float(box_xyxy[1]),
                "prompt_box_x1": float(box_xyxy[2]),
                "prompt_box_y1": float(box_xyxy[3]),
                "num_annotation_boxes": len(bboxes),
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
        },
        "intervention": {
            "type": "sam_background_suppression",
            "selection_strategy": "sam_predictor_with_annotation_bbox_prompt_else_central_box_prompt",
            "blur_radius": args.blur_radius,
            "mask_feather": args.mask_feather,
            "central_box_scale": args.central_box_scale,
            "min_mask_area_fraction": args.min_mask_area_fraction,
            "annotation_prompt_count": annotation_prompt_count,
            "fallback_central_box_count": fallback_count,
        },
    }
    save_json(variant_root / "metadata" / "background_intervention.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
