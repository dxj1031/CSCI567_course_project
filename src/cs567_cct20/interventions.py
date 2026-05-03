from __future__ import annotations

import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


TRAIN_INTERVENTIONS = {
    "none",
    "photometric_randomization",
    "background_perturbation",
    "combined",
    # Legacy names are retained so older configs remain inspectable, but the
    # active experiment matrix no longer submits these information-suppression variants.
    "bbox_blur",
    "brightness_aligned",
}
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


def train_lookup_keys(train_frame: pd.DataFrame) -> set[str]:
    lookup_keys: set[str] = set()
    for file_name in train_frame["file_name"].astype(str).tolist():
        path = Path(file_name)
        lookup_keys.add(file_name)
        lookup_keys.add(path.name)
        lookup_keys.add(path.stem)
    return lookup_keys


def parse_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number, number
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        return (lo, hi) if lo <= hi else (hi, lo)
    raise ValueError(f"Expected a scalar or two-value range, got: {value!r}")


def apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    if gamma <= 0:
        return image
    lookup = [int(np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255)) for value in range(256)]
    return image.point(lookup * len(image.getbands()))


def add_rgb_noise(image: Image.Image, std_fraction: float) -> Image.Image:
    if std_fraction <= 0:
        return image
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=255.0 * std_fraction, size=pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


class PhotometricRandomizationIntervention(TrainImageIntervention):
    name = "photometric_randomization"

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.65, 1.45),
        contrast_range: tuple[float, float] = (0.70, 1.35),
        gamma_range: tuple[float, float] = (0.70, 1.45),
        saturation_range: tuple[float, float] = (0.65, 1.45),
        noise_std_range: tuple[float, float] = (0.0, 0.025),
        transform_probability: float = 0.90,
        per_channel_gain_range: tuple[float, float] = (0.90, 1.10),
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.saturation_range = saturation_range
        self.noise_std_range = noise_std_range
        self.transform_probability = transform_probability
        self.per_channel_gain_range = per_channel_gain_range

    @staticmethod
    def sample_factor(value_range: tuple[float, float]) -> float:
        return random.uniform(value_range[0], value_range[1])

    def maybe_apply(self, image: Image.Image, enhancer_cls: type[ImageEnhance._Enhance], value_range: tuple[float, float]) -> Image.Image:
        if random.random() > self.transform_probability:
            return image
        return enhancer_cls(image).enhance(self.sample_factor(value_range))

    def apply_channel_gain(self, image: Image.Image) -> Image.Image:
        if random.random() > self.transform_probability:
            return image
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        gains = np.asarray(
            [self.sample_factor(self.per_channel_gain_range) for _ in range(3)],
            dtype=np.float32,
        )
        pixels = np.clip(pixels * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels, mode="RGB")

    def __call__(self, image: Image.Image, row: pd.Series) -> Image.Image:
        del row
        image = image.convert("RGB")
        operations = [
            lambda img: self.maybe_apply(img, ImageEnhance.Brightness, self.brightness_range),
            lambda img: self.maybe_apply(img, ImageEnhance.Contrast, self.contrast_range),
            lambda img: self.maybe_apply(img, ImageEnhance.Color, self.saturation_range),
            self.apply_channel_gain,
            lambda img: apply_gamma(img, self.sample_factor(self.gamma_range))
            if random.random() <= self.transform_probability
            else img,
            lambda img: add_rgb_noise(img, self.sample_factor(self.noise_std_range))
            if random.random() <= self.transform_probability
            else img,
        ]
        random.shuffle(operations)
        for operation in operations:
            image = operation(image)
        return image

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": "train_only",
            "strategy": "stochastic_distribution_diversification",
            "fit_statistics_source": "none",
            "applies_to_validation_or_test": False,
            "brightness_range": list(self.brightness_range),
            "contrast_range": list(self.contrast_range),
            "gamma_range": list(self.gamma_range),
            "saturation_range": list(self.saturation_range),
            "noise_std_range": list(self.noise_std_range),
            "per_channel_gain_range": list(self.per_channel_gain_range),
            "transform_probability": self.transform_probability,
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
        self.train_lookup_keys = train_lookup_keys(train_frame)
        raw_bbox_index = build_bbox_index(dataset_root)
        self.bbox_index = {
            key: records
            for key, records in raw_bbox_index.items()
            if key in self.train_lookup_keys
        }

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


class BackgroundPerturbationIntervention(TrainImageIntervention):
    name = "background_perturbation"

    def __init__(
        self,
        dataset_root: Path,
        train_frame: pd.DataFrame,
        blur_radius_range: tuple[float, float] = (0.20, 1.25),
        noise_std_range: tuple[float, float] = (0.0, 0.018),
        contrast_range: tuple[float, float] = (0.90, 1.10),
        operation_probability: float = 0.85,
        box_feather: float = 0.75,
        bbox_padding_fraction: float = 0.03,
    ) -> None:
        self.dataset_root = dataset_root
        self.blur_radius_range = blur_radius_range
        self.noise_std_range = noise_std_range
        self.contrast_range = contrast_range
        self.operation_probability = operation_probability
        self.box_feather = box_feather
        self.bbox_padding_fraction = bbox_padding_fraction
        self.train_lookup_keys = train_lookup_keys(train_frame)
        raw_bbox_index = build_bbox_index(dataset_root)
        self.bbox_index = {
            key: records
            for key, records in raw_bbox_index.items()
            if key in self.train_lookup_keys
        }

    @staticmethod
    def sample_factor(value_range: tuple[float, float]) -> float:
        return random.uniform(value_range[0], value_range[1])

    def perturb_background_candidate(self, image: Image.Image) -> Image.Image:
        perturbed = image.convert("RGB")
        if random.random() <= self.operation_probability:
            perturbed = perturbed.filter(
                ImageFilter.GaussianBlur(radius=self.sample_factor(self.blur_radius_range))
            )
        if random.random() <= self.operation_probability:
            perturbed = ImageEnhance.Contrast(perturbed).enhance(self.sample_factor(self.contrast_range))
        if random.random() <= self.operation_probability:
            perturbed = add_rgb_noise(perturbed, self.sample_factor(self.noise_std_range))
        return perturbed

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
        perturbed = self.perturb_background_candidate(image)
        mask_image = Image.fromarray((foreground_mask.astype(np.uint8) * 255), mode="L")
        if self.box_feather > 0:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=self.box_feather))
        return Image.composite(image, perturbed, mask_image)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": "train_only",
            "strategy": "structure_preserving_background_diversification",
            "fit_statistics_source": "training_annotations_only_for_training_samples",
            "applies_to_validation_or_test": False,
            "dataset_root": str(self.dataset_root),
            "blur_radius_range": list(self.blur_radius_range),
            "noise_std_range": list(self.noise_std_range),
            "contrast_range": list(self.contrast_range),
            "operation_probability": self.operation_probability,
            "box_feather": self.box_feather,
            "bbox_padding_fraction": self.bbox_padding_fraction,
            "train_lookup_key_count": len(self.train_lookup_keys),
            "bbox_index_key_count": len(self.bbox_index),
            "preserved_region": "scaled_annotation_bbox_with_padding",
        }


class CombinedDiversificationIntervention(TrainImageIntervention):
    name = "combined"

    def __init__(
        self,
        photometric: PhotometricRandomizationIntervention,
        background: BackgroundPerturbationIntervention,
    ) -> None:
        self.photometric = photometric
        self.background = background

    def __call__(self, image: Image.Image, row: pd.Series) -> Image.Image:
        image = self.background(image, row)
        return self.photometric(image, row)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": "train_only",
            "strategy": "photometric_randomization_plus_background_perturbation",
            "fit_statistics_source": "training_annotations_only_for_background_boxes",
            "applies_to_validation_or_test": False,
            "components": {
                "photometric_randomization": self.photometric.summary(),
                "background_perturbation": self.background.summary(),
            },
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
    if intervention_name == "photometric_randomization":
        photo_params = params.get("photometric_randomization", params)
        return PhotometricRandomizationIntervention(
            brightness_range=parse_range(photo_params.get("brightness_range"), (0.65, 1.45)),
            contrast_range=parse_range(photo_params.get("contrast_range"), (0.70, 1.35)),
            gamma_range=parse_range(photo_params.get("gamma_range"), (0.70, 1.45)),
            saturation_range=parse_range(photo_params.get("saturation_range"), (0.65, 1.45)),
            noise_std_range=parse_range(photo_params.get("noise_std_range"), (0.0, 0.025)),
            transform_probability=float(photo_params.get("transform_probability", 0.90)),
            per_channel_gain_range=parse_range(photo_params.get("per_channel_gain_range"), (0.90, 1.10)),
        )
    if intervention_name == "background_perturbation":
        background_params = params.get("background_perturbation", params)
        return BackgroundPerturbationIntervention(
            dataset_root=dataset_root,
            train_frame=train_frame,
            blur_radius_range=parse_range(background_params.get("blur_radius_range"), (0.20, 1.25)),
            noise_std_range=parse_range(background_params.get("noise_std_range"), (0.0, 0.018)),
            contrast_range=parse_range(background_params.get("contrast_range"), (0.90, 1.10)),
            operation_probability=float(background_params.get("operation_probability", 0.85)),
            box_feather=float(background_params.get("box_feather", 0.75)),
            bbox_padding_fraction=float(background_params.get("bbox_padding_fraction", 0.03)),
        )
    if intervention_name == "combined":
        combined_params = params.get("combined", params)
        return CombinedDiversificationIntervention(
            photometric=build_train_intervention(
                name="photometric_randomization",
                train_frame=train_frame,
                images_dir=images_dir,
                dataset_root=dataset_root,
                params=combined_params.get("photometric_randomization", combined_params),
            ),
            background=build_train_intervention(
                name="background_perturbation",
                train_frame=train_frame,
                images_dir=images_dir,
                dataset_root=dataset_root,
                params=combined_params.get("background_perturbation", combined_params),
            ),
        )
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
