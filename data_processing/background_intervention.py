#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import shutil
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


@dataclass(frozen=True)
class BBoxRecord:
    bbox_xywh: tuple[float, float, float, float]
    annotation_width: float | None
    annotation_height: float | None


@dataclass(frozen=True)
class ScaledBox:
    xyxy: tuple[int, int, int, int]
    bbox_xywh: tuple[float, float, float, float]
    annotation_size: tuple[float | None, float | None]
    scale_xy: tuple[float, float]


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
        help="Expand each scaled annotation box by this fraction of the larger loaded image side before preserving it.",
    )
    return parser.parse_args()


def positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def scale_xywh_to_xyxy(
    record: BBoxRecord,
    target_width: int,
    target_height: int,
    padding: float,
) -> ScaledBox:
    x, y, w, h = record.bbox_xywh
    source_width = record.annotation_width or float(target_width)
    source_height = record.annotation_height or float(target_height)
    scale_x = float(target_width) / source_width
    scale_y = float(target_height) / source_height

    scaled_x0 = x * scale_x
    scaled_y0 = y * scale_y
    scaled_x1 = (x + w) * scale_x
    scaled_y1 = (y + h) * scale_y

    pad = padding * float(max(target_width, target_height))
    x0 = max(0, int(np.floor(scaled_x0 - pad)))
    y0 = max(0, int(np.floor(scaled_y0 - pad)))
    x1 = min(target_width, int(np.ceil(scaled_x1 + pad)))
    y1 = min(target_height, int(np.ceil(scaled_y1 + pad)))
    if x1 <= x0:
        x1 = min(target_width, x0 + 1)
    if y1 <= y0:
        y1 = min(target_height, y0 + 1)

    return ScaledBox(
        xyxy=(x0, y0, x1, y1),
        bbox_xywh=record.bbox_xywh,
        annotation_size=(record.annotation_width, record.annotation_height),
        scale_xy=(scale_x, scale_y),
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


def build_bbox_index(source_root: Path) -> dict[str, list[BBoxRecord]]:
    payloads = load_annotation_payloads(source_root)
    bbox_index: dict[str, list[BBoxRecord]] = {}

    for payload in payloads:
        image_id_to_metadata: dict[str, dict[str, Any]] = {}
        for image in payload.get("images", []):
            image_id = str(image.get("id", ""))
            file_name = str(image.get("file_name", ""))
            if image_id:
                image_id_to_metadata[image_id] = {
                    "file_name": file_name,
                    "width": positive_float_or_none(image.get("width")),
                    "height": positive_float_or_none(image.get("height")),
                }

        for annotation in payload.get("annotations", []):
            image_id = str(annotation.get("image_id", ""))
            bbox = annotation.get("bbox")
            if not image_id or bbox is None:
                continue
            if len(bbox) != 4:
                continue
            bbox_values = tuple(float(value) for value in bbox)
            image_metadata = image_id_to_metadata.get(image_id, {})
            record = BBoxRecord(
                bbox_xywh=bbox_values,
                annotation_width=image_metadata.get("width"),
                annotation_height=image_metadata.get("height"),
            )
            lookup_keys = {image_id}
            file_name = str(image_metadata.get("file_name") or "")
            if file_name:
                lookup_keys.add(Path(file_name).stem)
                lookup_keys.add(Path(file_name).name)
            for lookup_key in lookup_keys:
                bbox_index.setdefault(lookup_key, []).append(record)

    return bbox_index


def build_bbox_mask(
    bboxes: list[BBoxRecord],
    width: int,
    height: int,
    padding: float,
) -> tuple[np.ndarray, list[ScaledBox]]:
    mask = np.zeros((height, width), dtype=bool)
    scaled_boxes: list[ScaledBox] = []
    for bbox in bboxes:
        scaled_box = scale_xywh_to_xyxy(bbox, width, height, padding)
        x0, y0, x1, y1 = scaled_box.xyxy
        mask[y0:y1, x0:x1] = True
        scaled_boxes.append(scaled_box)
    return mask, scaled_boxes


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
    # White mask pixels select the original image; black pixels select the blurred background.
    return Image.composite(image, blurred, mask_image)


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def copy_original_image(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def build_file_lookup_keys(file_names: set[str]) -> set[str]:
    lookup_keys: set[str] = set()
    for file_name in file_names:
        path = Path(file_name)
        lookup_keys.add(file_name)
        lookup_keys.add(path.name)
        lookup_keys.add(path.stem)
    return lookup_keys


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
    train_files = set(tables["cct20_train"]["file_name"]) if "cct20_train" in tables else set()
    train_lookup_keys = build_file_lookup_keys(train_files)
    bbox_index = {
        key: records
        for key, records in bbox_index.items()
        if key in train_lookup_keys
    }

    selection_rows: list[dict[str, Any]] = []
    missing_bbox_count = 0
    annotation_bbox_count = 0
    train_transformed_count = 0
    non_train_unchanged_count = 0

    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        output_path = output_images_dir / row.file_name
        image_id = Path(row.file_name).stem
        is_train_image = row.file_name in train_files
        copied_original_file = False

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            width, height = image.size

            bboxes = bbox_index.get(image_id, [])
            if not is_train_image:
                foreground_mask = np.ones((height, width), dtype=bool)
                scaled_boxes = []
                selection_source = "non_train_image_unchanged"
                transformed = image
                copied_original_file = True
                non_train_unchanged_count += 1
            elif bboxes:
                foreground_mask, scaled_boxes = build_bbox_mask(
                    bboxes=bboxes,
                    width=width,
                    height=height,
                    padding=args.bbox_padding_fraction,
                )
                selection_source = "train_annotation_bbox"
                annotation_bbox_count += 1
                train_transformed_count += 1
                transformed = apply_background_suppression(
                    image=image,
                    foreground_mask=foreground_mask,
                    blur_radius=args.blur_radius,
                    box_feather=args.box_feather,
                )
            else:
                foreground_mask = np.ones((height, width), dtype=bool)
                scaled_boxes = []
                selection_source = "train_missing_annotation_bbox_image_unchanged"
                transformed = image
                copied_original_file = True
                missing_bbox_count += 1

            if copied_original_file:
                copy_original_image(input_path, output_path)
            else:
                save_image(transformed, output_path)

        selection_rows.append(
            {
                "file_name": row.file_name,
                "split": getattr(row, "split", None),
                "day_night": getattr(row, "day_night", None),
                "selection_source": selection_source,
                "is_train_image": bool(is_train_image),
                "was_transformed": selection_source == "train_annotation_bbox",
                "original_file_copied": copied_original_file,
                "foreground_area_fraction": float(foreground_mask.mean()),
                "boxes_xyxy": json.dumps([box.xyxy for box in scaled_boxes]),
                "annotation_boxes_xywh": json.dumps([box.bbox_xywh for box in scaled_boxes]),
                "annotation_image_sizes": json.dumps([box.annotation_size for box in scaled_boxes]),
                "bbox_scale_factors": json.dumps([box.scale_xy for box in scaled_boxes]),
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
            "selection_strategy": "train_only_scale_annotation_bboxes_preserve_inside_blur_outside",
            "transform_scope": "train_only",
            "evaluation_scope": "validation_and_test_images_left_original",
            "coordinate_transform": (
                "bbox xywh is scaled from annotation image size to the loaded image size; "
                "for training images, original loaded pixels inside scaled boxes are preserved and pixels outside are blurred"
            ),
            "blur_radius": args.blur_radius,
            "box_feather": args.box_feather,
            "bbox_padding_fraction": args.bbox_padding_fraction,
            "annotation_source_scope": "bbox records filtered to training file names before transformation",
            "train_lookup_key_count": len(train_lookup_keys),
            "bbox_index_key_count": len(bbox_index),
            "train_transformed_count": train_transformed_count,
            "non_train_unchanged_count": non_train_unchanged_count,
            "annotation_bbox_count": annotation_bbox_count,
            "missing_bbox_count": missing_bbox_count,
        },
    }
    save_json(variant_root / "metadata" / "background_intervention.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
