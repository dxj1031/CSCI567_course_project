from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from .interventions import BBoxRecord, build_bbox_index, build_bbox_mask


IMAGE_ABLATIONS = {"none", "object_crop", "foreground_only", "background_only"}


@dataclass(frozen=True)
class ImageAblation:
    name: str = "none"
    bbox_index: dict[str, list[BBoxRecord]] | None = None
    bbox_padding_fraction: float = 0.04
    mask_feather: float = 2.0
    fill_color: tuple[int, int, int] = (127, 127, 127)
    missing_bbox_policy: str = "keep_original"

    def __call__(self, image: Image.Image, row: pd.Series) -> Image.Image:
        if self.name == "none":
            return image

        records = self.lookup_records(str(row["file_name"]))
        if not records:
            return image

        image = image.convert("RGB")
        if self.name == "object_crop":
            return self.object_crop(image, records)
        if self.name in {"foreground_only", "background_only"}:
            return self.masked_view(image, records)
        raise AssertionError(f"Unhandled image ablation: {self.name}")

    def lookup_records(self, file_name: str) -> list[BBoxRecord]:
        if self.bbox_index is None:
            return []
        path = Path(file_name)
        return (
            self.bbox_index.get(file_name)
            or self.bbox_index.get(path.name)
            or self.bbox_index.get(path.stem)
            or []
        )

    def object_crop(self, image: Image.Image, records: list[BBoxRecord]) -> Image.Image:
        _, boxes = build_bbox_mask(
            records=records,
            width=image.width,
            height=image.height,
            padding_fraction=self.bbox_padding_fraction,
        )
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        return image.crop((x0, y0, x1, y1))

    def masked_view(self, image: Image.Image, records: list[BBoxRecord]) -> Image.Image:
        foreground_mask, _ = build_bbox_mask(
            records=records,
            width=image.width,
            height=image.height,
            padding_fraction=self.bbox_padding_fraction,
        )
        mask = Image.fromarray((foreground_mask.astype(np.uint8) * 255), mode="L")
        if self.mask_feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.mask_feather))

        fill = Image.new("RGB", image.size, self.fill_color)
        if self.name == "foreground_only":
            return Image.composite(image, fill, mask)
        if self.name == "background_only":
            return Image.composite(fill, image, mask)
        raise AssertionError(f"Unhandled masked ablation: {self.name}")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": "train_validation_test_consistent",
            "annotation_source": "all_available_cct20_annotation_files",
            "bbox_index_key_count": 0 if self.bbox_index is None else len(self.bbox_index),
            "bbox_padding_fraction": self.bbox_padding_fraction,
            "mask_feather": self.mask_feather,
            "fill_color": list(self.fill_color),
            "missing_bbox_policy": self.missing_bbox_policy,
        }


def parse_fill_color(value: Any, default: tuple[int, int, int] = (127, 127, 127)) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Expected fill color like '127,127,127', got {value!r}.")
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(part) for part in value)  # type: ignore[return-value]
    raise ValueError(f"Expected fill color as a 3-value RGB tuple, got {value!r}.")


def build_image_ablation(
    name: str,
    dataset_root: Path,
    params: dict[str, Any] | None = None,
) -> ImageAblation:
    ablation_name = name.lower()
    if ablation_name not in IMAGE_ABLATIONS:
        raise ValueError(f"Unsupported image ablation: {name}. Expected one of {sorted(IMAGE_ABLATIONS)}")

    params = params or {}
    if ablation_name == "none":
        return ImageAblation()

    return ImageAblation(
        name=ablation_name,
        bbox_index=build_bbox_index(dataset_root),
        bbox_padding_fraction=float(params.get("bbox_padding_fraction", 0.04)),
        mask_feather=float(params.get("mask_feather", 2.0)),
        fill_color=parse_fill_color(params.get("fill_color")),
        missing_bbox_policy=str(params.get("missing_bbox_policy", "keep_original")),
    )
