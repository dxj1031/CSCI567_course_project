from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


VISIBILITY_MODES = {"original", "gamma", "clahe", "gamma_clahe"}
VISIBILITY_SCOPES = {"test_only", "train_test_consistent", "night_only"}
DEFAULT_NIGHT_VALUES = {"night", "low_light", "low-light", "dark", "1", "true", "yes"}


def apply_gamma_correction(image: Image.Image, gamma: float) -> Image.Image:
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma}.")
    rgb_image = image.convert("RGB")
    lookup = [
        int(np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255))
        for value in range(256)
    ]
    return rgb_image.point(lookup * len(rgb_image.getbands()))


def _clip_histogram(histogram: np.ndarray, clip_limit: float, tile_pixel_count: int) -> np.ndarray:
    if clip_limit <= 0:
        return histogram

    clip_threshold = max(int(clip_limit * tile_pixel_count / 256), 1)
    clipped = np.minimum(histogram, clip_threshold)
    excess = int(histogram.sum() - clipped.sum())
    if excess <= 0:
        return clipped

    clipped += excess // 256
    remainder = excess % 256
    if remainder:
        clipped[:remainder] += 1
    return clipped


def _build_lut(values: np.ndarray, clip_limit: float) -> np.ndarray:
    histogram = np.bincount(values.reshape(-1), minlength=256).astype(np.int64)
    histogram = _clip_histogram(histogram, clip_limit=clip_limit, tile_pixel_count=values.size)
    cdf = histogram.cumsum()
    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0:
        return np.arange(256, dtype=np.uint8)
    cdf_min = int(nonzero[0])
    denominator = int(cdf[-1] - cdf_min)
    if denominator <= 0:
        return np.arange(256, dtype=np.uint8)
    lut = np.rint((cdf - cdf_min) * 255.0 / denominator)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _apply_numpy_clahe_to_channel(
    channel: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    tiles_x, tiles_y = tile_grid_size
    if tiles_x <= 0 or tiles_y <= 0:
        raise ValueError(f"tile_grid_size must contain positive integers, got {tile_grid_size}.")

    height, width = channel.shape
    tile_width = int(np.ceil(width / tiles_x))
    tile_height = int(np.ceil(height / tiles_y))
    output = np.empty_like(channel)

    for tile_y in range(tiles_y):
        y0 = tile_y * tile_height
        y1 = min(height, y0 + tile_height)
        if y0 >= y1:
            continue
        for tile_x in range(tiles_x):
            x0 = tile_x * tile_width
            x1 = min(width, x0 + tile_width)
            if x0 >= x1:
                continue
            tile = channel[y0:y1, x0:x1]
            lut = _build_lut(tile, clip_limit=clip_limit)
            output[y0:y1, x0:x1] = lut[tile]

    return output


def _apply_numpy_clahe(
    image: Image.Image,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> Image.Image:
    ycbcr = np.asarray(image.convert("RGB").convert("YCbCr"), dtype=np.uint8).copy()
    ycbcr[:, :, 0] = _apply_numpy_clahe_to_channel(
        ycbcr[:, :, 0],
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )
    return Image.fromarray(ycbcr, mode="YCbCr").convert("RGB")


def apply_clahe(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}.")

    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _apply_numpy_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb, mode="RGB")


def apply_gamma_then_clahe(
    image: Image.Image,
    gamma: float = 0.7,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    return apply_clahe(
        apply_gamma_correction(image, gamma=gamma),
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )


def parse_tile_grid_size(value: Any, default: tuple[int, int] = (8, 8)) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.lower().replace("x", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected tile grid size like '8,8' or '8x8', got {value!r}.")
        return int(parts[0]), int(parts[1])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected tile grid size like '8,8' or [8, 8], got {value!r}.")


def _normalize_metadata_value(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _parse_night_rule(source: str) -> tuple[str, set[str]] | None:
    if "=" not in source:
        return None
    field_name, expected_values = source.split("=", 1)
    values = {
        _normalize_metadata_value(value)
        for value in expected_values.replace("|", ",").split(",")
        if value.strip()
    }
    if not field_name.strip() or not values:
        raise ValueError(f"Invalid night metadata rule: {source!r}. Use a form like 'day_night=night'.")
    return field_name.strip(), values


@dataclass(frozen=True)
class VisibilitySettings:
    mode: str = "original"
    scope: str = "train_test_consistent"
    night_only_flag_source: str = "day_night"
    gamma: float = 0.7
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    experiment_requested: bool = False

    def __post_init__(self) -> None:
        if self.mode not in VISIBILITY_MODES:
            raise ValueError(f"Unsupported visibility mode: {self.mode}. Expected one of {sorted(VISIBILITY_MODES)}.")
        if self.scope not in VISIBILITY_SCOPES:
            raise ValueError(
                f"Unsupported visibility scope: {self.scope}. Expected one of {sorted(VISIBILITY_SCOPES)}."
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "original"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scope": self.scope,
            "night_only_flag_source": self.night_only_flag_source,
            "gamma": self.gamma,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid_size": list(self.clahe_tile_grid_size),
            "experiment_requested": self.experiment_requested,
            "enabled": self.enabled,
        }


class VisibilityPreprocessor:
    def __init__(self, settings: VisibilitySettings) -> None:
        self.settings = settings
        self.name = f"visibility_{settings.scope}_{settings.mode}"

    def is_night_row(self, row: pd.Series) -> bool:
        rule = _parse_night_rule(self.settings.night_only_flag_source)
        if rule is None:
            field_name = self.settings.night_only_flag_source
            expected_values = DEFAULT_NIGHT_VALUES
        else:
            field_name, expected_values = rule

        if field_name not in row:
            return False
        return _normalize_metadata_value(row.get(field_name)) in expected_values

    def should_apply(self, split_name: str, row: pd.Series) -> bool:
        if not self.settings.enabled:
            return False
        if self.settings.scope == "train_test_consistent":
            return True
        if self.settings.scope == "test_only":
            return split_name != "train" and self.is_night_row(row)
        if self.settings.scope == "night_only":
            return self.is_night_row(row)
        raise AssertionError(f"Unhandled visibility scope: {self.settings.scope}")

    def __call__(self, image: Image.Image, row: pd.Series, split_name: str) -> Image.Image:
        if not self.should_apply(split_name, row):
            return image
        if self.settings.mode == "gamma":
            return apply_gamma_correction(image, gamma=self.settings.gamma)
        if self.settings.mode == "clahe":
            return apply_clahe(
                image,
                clip_limit=self.settings.clahe_clip_limit,
                tile_grid_size=self.settings.clahe_tile_grid_size,
            )
        if self.settings.mode == "gamma_clahe":
            return apply_gamma_then_clahe(
                image,
                gamma=self.settings.gamma,
                clip_limit=self.settings.clahe_clip_limit,
                tile_grid_size=self.settings.clahe_tile_grid_size,
            )
        return image

    def summary(self, dataframes: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
        payload = self.settings.to_dict()
        payload["name"] = self.name
        payload["applies_to_training"] = bool(
            self.settings.enabled and self.settings.scope in {"train_test_consistent", "night_only"}
        )
        payload["applies_to_validation_or_test"] = self.settings.enabled
        payload["application_counts"] = (
            count_visibility_applications(dataframes, self) if dataframes is not None else {}
        )
        return payload


def build_visibility_settings(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> VisibilitySettings:
    visibility_cfg = dict(config.get("visibility", {}) or {})
    overrides = overrides or {}
    for key, value in overrides.items():
        if value is not None:
            visibility_cfg[key] = value

    tile_grid_size = parse_tile_grid_size(visibility_cfg.get("clahe_tile_grid_size"), default=(8, 8))
    return VisibilitySettings(
        mode=str(visibility_cfg.get("mode", "original")).lower(),
        scope=str(visibility_cfg.get("scope", "train_test_consistent")).lower(),
        night_only_flag_source=str(visibility_cfg.get("night_only_flag_source", "day_night")),
        gamma=float(visibility_cfg.get("gamma", 0.7)),
        clahe_clip_limit=float(visibility_cfg.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=tile_grid_size,
        experiment_requested=bool(visibility_cfg.get("experiment_requested", False)),
    )


def count_visibility_applications(
    dataframes: dict[str, pd.DataFrame],
    preprocessor: VisibilityPreprocessor,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split_name, frame in dataframes.items():
        applied = 0
        night_rows = 0
        for _, row in frame.iterrows():
            if preprocessor.is_night_row(row):
                night_rows += 1
            if preprocessor.should_apply(split_name, row):
                applied += 1
        counts[split_name] = {
            "rows": int(len(frame)),
            "night_rows": int(night_rows),
            "enhanced_rows": int(applied),
        }
    return counts
