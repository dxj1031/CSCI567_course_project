from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


TRAIN_INTERVENTIONS = {"none", "bbox_blur", "brightness_aligned"}
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


def positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def load_annotation_payloads(dataset_root: Path) -> list[dict[str, Any]]:
    annotations_dir = dataset_root / "annotations"
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
                if extracted is not None:
                    payloads.append(json.load(extracted))
        if payloads:
            return payloads

    for json_name in ANNOTATION_JSON_NAMES:
        direct_path = annotations_dir / json_name
        if direct_path.exists():
            payloads.append(json.loads(direct_path.read_text(encoding="utf-8")))

    return payloads


def build_bbox_index(dataset_root: Path) -> dict[str, list[BBoxRecord]]:
    bbox_index: dict[str, list[BBoxRecord]] = {}
    for payload in load_annotation_payloads(dataset_root):
        image_metadata_by_id: dict[str, dict[str, Any]] = {}
        for image in payload.get("images", []):
            image_id = str(image.get("id", ""))
            if not image_id:
                continue
            image_metadata_by_id[image_id] = {
                "file_name": str(image.get("file_name", "")),
                "width": positive_float_or_none(image.get("width")),
                "height": positive_float_or_none(image.get("height")),
            }

        for annotation in payload.get("annotations", []):
            image_id = str(annotation.get("image_id", ""))
            bbox = annotation.get("bbox")
            if not image_id or bbox is None or len(bbox) != 4:
                continue

            image_metadata = image_metadata_by_id.get(image_id, {})
            record = BBoxRecord(
                bbox_xywh=tuple(float(value) for value in bbox),
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


def scale_xywh_to_xyxy(
    record: BBoxRecord,
    target_width: int,
    target_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = record.bbox_xywh
    source_width = record.annotation_width or float(target_width)
    source_height = record.annotation_height or float(target_height)
    scale_x = float(target_width) / source_width
    scale_y = float(target_height) / source_height
    pad = padding_fraction * float(max(target_width, target_height))

    x0 = max(0, int(np.floor(x * scale_x - pad)))
    y0 = max(0, int(np.floor(y * scale_y - pad)))
    x1 = min(target_width, int(np.ceil((x + w) * scale_x + pad)))
    y1 = min(target_height, int(np.ceil((y + h) * scale_y + pad)))
    if x1 <= x0:
        x1 = min(target_width, x0 + 1)
    if y1 <= y0:
        y1 = min(target_height, y0 + 1)
    return x0, y0, x1, y1


def build_bbox_mask(
    records: list[BBoxRecord],
    width: int,
    height: int,
    padding_fraction: float,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    mask = np.zeros((height, width), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for record in records:
        box = scale_xywh_to_xyxy(record, width, height, padding_fraction)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        boxes.append(box)
    return mask, boxes


class TrainImageIntervention:
    name = "none"

    def __call__(self, image: Image.Image, row: pd.Series) -> Image.Image:
        return image

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": "train_only",
            "fit_statistics_source": "none",
            "applies_to_validation_or_test": False,
        }


class BBoxBlurIntervention(TrainImageIntervention):
    name = "bbox_blur"

    def __init__(
        self,
        dataset_root: Path,
        train_frame: pd.DataFrame,
        blur_radius: float = 8.0,
        box_feather: float = 3.0,
        bbox_padding_fraction: float = 0.02,
    ) -> None:
        self.dataset_root = dataset_root
        self.blur_radius = blur_radius
        self.box_feather = box_feather
        self.bbox_padding_fraction = bbox_padding_fraction
        self.train_lookup_keys = self.build_train_lookup_keys(train_frame)
        raw_bbox_index = build_bbox_index(dataset_root)
        self.bbox_index = {
            key: records
            for key, records in raw_bbox_index.items()
            if key in self.train_lookup_keys
        }

    @staticmethod
    def build_train_lookup_keys(train_frame: pd.DataFrame) -> set[str]:
        lookup_keys: set[str] = set()
        for file_name in train_frame["file_name"].astype(str).tolist():
            path = Path(file_name)
            lookup_keys.add(file_name)
            lookup_keys.add(path.name)
            lookup_keys.add(path.stem)
        return lookup_keys

    def __call__(self, image: Image.Image, row: pd.Series) -> Image.Image:
        file_name = str(row["file_name"])
        records = self.bbox_index.get(Path(file_name).stem) or self.bbox_index.get(file_name) or []
        if not records:
            return image

        image = image.convert("RGB")
        foreground_mask, _ = build_bbox_mask(
            records=records,
            width=image.width,
            height=image.height,
            padding_fraction=self.bbox_padding_fraction,
        )
        blurred = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        mask_image = Image.fromarray((foreground_mask.astype(np.uint8) * 255), mode="L")
        if self.box_feather > 0:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=self.box_feather))
        return Image.composite(image, blurred, mask_image)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": "train_only",
            "fit_statistics_source": "training_annotations_only_for_training_samples",
            "applies_to_validation_or_test": False,
            "dataset_root": str(self.dataset_root),
            "blur_radius": self.blur_radius,
            "box_feather": self.box_feather,
            "bbox_padding_fraction": self.bbox_padding_fraction,
            "train_lookup_key_count": len(self.train_lookup_keys),
            "bbox_index_key_count": len(self.bbox_index),
        }


def extract_value_channel(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("RGB").convert("HSV"), dtype=np.uint8)
    return hsv[:, :, 2]


def accumulate_histogram(histogram: np.ndarray, values: np.ndarray) -> None:
    histogram += np.bincount(values.reshape(-1), minlength=256)


def build_lookup_table(source_hist: np.ndarray, target_hist: np.ndarray) -> np.ndarray:
    if source_hist.sum() == 0 or target_hist.sum() == 0:
        return np.arange(256, dtype=np.uint8)

    source_cdf = np.cumsum(source_hist, dtype=np.float64)
    target_cdf = np.cumsum(target_hist, dtype=np.float64)
    source_cdf /= source_cdf[-1]
    target_cdf /= target_cdf[-1]

    mapped = np.interp(source_cdf, target_cdf, np.arange(256, dtype=np.float64))
    return np.clip(np.rint(mapped), 0, 255).astype(np.uint8)


def apply_histogram_lookup(image: Image.Image, lookup: np.ndarray) -> Image.Image:
    hsv = np.asarray(image.convert("RGB").convert("HSV"), dtype=np.uint8).copy()
    hsv[:, :, 2] = lookup[hsv[:, :, 2]]
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


class BrightnessAlignedIntervention(TrainImageIntervention):
    name = "brightness_aligned"

    def __init__(self, train_frame: pd.DataFrame, images_dir: Path) -> None:
        self.images_dir = images_dir
        self.train_histograms = {
            "day": np.zeros(256, dtype=np.int64),
            "night": np.zeros(256, dtype=np.int64),
        }

        for row in train_frame.itertuples(index=False):
            day_night = getattr(row, "day_night", None)
            if day_night not in self.train_histograms:
                continue
            image_path = images_dir / getattr(row, "file_name")
            with Image.open(image_path) as image:
                accumulate_histogram(self.train_histograms[day_night], extract_value_channel(image))

        target_histogram = self.train_histograms["day"] + self.train_histograms["night"]
        self.lookup_tables = {
            group_name: build_lookup_table(group_hist, target_histogram)
            for group_name, group_hist in self.train_histograms.items()
        }

    def __call__(self, image: Image.Image, row: pd.Series) -> Image.Image:
        day_night = row.get("day_night")
        lookup = self.lookup_tables.get(day_night)
        if lookup is None:
            return image
        return apply_histogram_lookup(image, lookup)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": "train_only",
            "fit_statistics_source": "training_split_only",
            "applies_to_validation_or_test": False,
            "color_space": "HSV_value_channel",
            "target_distribution": "combined_train_day_and_night_histogram",
            "train_day_pixel_count": int(self.train_histograms["day"].sum()),
            "train_night_pixel_count": int(self.train_histograms["night"].sum()),
        }


def build_train_intervention(
    name: str,
    train_frame: pd.DataFrame,
    images_dir: Path,
    dataset_root: Path,
    params: dict[str, Any] | None = None,
) -> TrainImageIntervention:
    intervention_name = name.lower()
    if intervention_name not in TRAIN_INTERVENTIONS:
        raise ValueError(f"Unsupported train intervention: {name}. Expected one of {sorted(TRAIN_INTERVENTIONS)}")

    params = params or {}
    if intervention_name == "none":
        return TrainImageIntervention()
    if intervention_name == "bbox_blur":
        bbox_params = params.get("bbox_blur", params)
        return BBoxBlurIntervention(
            dataset_root=dataset_root,
            train_frame=train_frame,
            blur_radius=float(bbox_params.get("blur_radius", 8.0)),
            box_feather=float(bbox_params.get("box_feather", 3.0)),
            bbox_padding_fraction=float(bbox_params.get("bbox_padding_fraction", 0.02)),
        )
    if intervention_name == "brightness_aligned":
        return BrightnessAlignedIntervention(train_frame=train_frame, images_dir=images_dir)

    raise AssertionError(f"Unhandled intervention: {intervention_name}")
