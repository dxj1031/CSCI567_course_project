#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
        description="Create a bbox-based background-blurred CCT20 dataset variant."
    )
    parser.add_argument("--source-root", required=True, help="Original dataset root containing images/ and processed/.")
    parser.add_argument("--output-root", required=True, help="Directory where dataset variants should be stored.")
    parser.add_argument("--variant-name", default="dataset_bbox_bg", help="Name of the output dataset variant.")
    parser.add_argument("--blur-radius", type=float, default=8.0, help="Gaussian blur radius applied to the background.")
    parser.add_argument(
        "--box-feather",
        type=float,
        default=3.0,
        help="Blur radius applied to the bbox mask edges for smoother transitions.",
    )
    parser.add_argument(
        "--bbox-padding-fraction",
        type=float,
        default=0.02,
        help="Expand each annotation box by this fraction of the larger image side before preserving it.",
    )
    return parser.parse_args()


def xywh_to_xyxy(bbox: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad = padding * float(max(width, height))
    x0 = max(0, int(np.floor(x - pad)))
    y0 = max(0, int(np.floor(y - pad)))
    x1 = min(width, int(np.ceil(x + w + pad)))
    y1 = min(height, int(np.ceil(y + h + pad)))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


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
        image_id_to_file_names: dict[str, list[str]] = {}
        for image in payload.get("images", []):
            image_id = str(image.get("id", ""))
            file_name = str(image.get("file_name", ""))
            if image_id and file_name:
                image_id_to_file_names.setdefault(image_id, []).append(file_name)

        for annotation in payload.get("annotations", []):
            image_id = str(annotation.get("image_id", ""))
            bbox = annotation.get("bbox")
            if not image_id or bbox is None:
                continue
            bbox_values = [float(value) for value in bbox]
            lookup_keys = {image_id}
            for file_name in image_id_to_file_names.get(image_id, []):
                lookup_keys.add(Path(file_name).stem)
                lookup_keys.add(Path(file_name).name)
            for lookup_key in lookup_keys:
                bbox_index.setdefault(lookup_key, []).append(bbox_values)

    return bbox_index


def build_bbox_mask(
    bboxes: list[list[float]],
    width: int,
    height: int,
    padding: float,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    mask = np.zeros((height, width), dtype=bool)
    boxes_xyxy: list[tuple[int, int, int, int]] = []
    for bbox in bboxes:
        box = xywh_to_xyxy(bbox, width, height, padding)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        boxes_xyxy.append(box)
    return mask, boxes_xyxy


def apply_background_suppression(
    image: Image.Image,
    foreground_mask: np.ndarray,
    blur_radius: float,
    box_feather: float,
) -> Image.Image:
    image = image.convert("RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_image = Image.fromarray((foreground_mask.astype(np.uint8) * 255), mode="L")
    if box_feather > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=box_feather))
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

    tables = load_processed_tables(source_root)
    master = build_master_table(tables)
    source_images_dir = resolve_source_images_dir(source_root, master["file_name"].head(32).tolist())
    variant_root = build_variant_root(output_root, args.variant_name)
    copy_processed_metadata(source_root, variant_root)
    output_images_dir = variant_root / "images"

    bbox_index = build_bbox_index(source_root)

    selection_rows: list[dict[str, Any]] = []
    missing_bbox_count = 0
    annotation_bbox_count = 0

    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        output_path = output_images_dir / row.file_name
        image_id = Path(row.file_name).stem

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            width, height = image.size

            bboxes = bbox_index.get(image_id, [])
            if bboxes:
                foreground_mask, boxes_xyxy = build_bbox_mask(
                    bboxes=bboxes,
                    width=width,
                    height=height,
                    padding=args.bbox_padding_fraction,
                )
                selection_source = "annotation_bbox"
                annotation_bbox_count += 1
            else:
                foreground_mask = np.ones((height, width), dtype=bool)
                boxes_xyxy = []
                selection_source = "missing_annotation_bbox_image_unchanged"
                missing_bbox_count += 1

            transformed = apply_background_suppression(
                image=image,
                foreground_mask=foreground_mask,
                blur_radius=args.blur_radius,
                box_feather=args.box_feather,
            )
            save_image(transformed, output_path)

        selection_rows.append(
            {
                "file_name": row.file_name,
                "split": getattr(row, "split", None),
                "day_night": getattr(row, "day_night", None),
                "selection_source": selection_source,
                "foreground_area_fraction": float(foreground_mask.mean()),
                "boxes_xyxy": json.dumps(boxes_xyxy),
                "num_annotation_boxes": len(bboxes),
            }
        )

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(variant_root / "metadata" / "bbox_background_blur.csv", index=False)

    metadata = {
        "variant_name": args.variant_name,
        "source_root": str(source_root),
        "resolved_source_images_dir": str(source_images_dir),
        "variant_root": str(variant_root),
        "num_images_processed": int(len(master)),
        "intervention": {
            "type": "bbox_background_blur",
            "selection_strategy": "preserve_annotation_bboxes_blur_outside_boxes",
            "blur_radius": args.blur_radius,
            "box_feather": args.box_feather,
            "bbox_padding_fraction": args.bbox_padding_fraction,
            "annotation_bbox_count": annotation_bbox_count,
            "missing_bbox_count": missing_bbox_count,
        },
    }
    save_json(variant_root / "metadata" / "background_intervention.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
