#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

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
        description="Create a histogram-matched illumination-aligned CCT20 dataset variant."
    )
    parser.add_argument("--source-root", required=True, help="Original dataset root containing images/ and processed/.")
    parser.add_argument("--output-root", required=True, help="Directory where dataset variants should be stored.")
    parser.add_argument("--variant-name", default="dataset_histmatch", help="Name of the output dataset variant.")
    parser.add_argument(
        "--target-mode",
        default="combined_train",
        choices=["combined_train"],
        help="Brightness target distribution strategy. Currently only the combined train distribution is supported.",
    )
    return parser.parse_args()


def extract_value_channel(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("RGB").convert("HSV"), dtype=np.uint8)
    return hsv[:, :, 2]


def compute_brightness(image: Image.Image) -> float:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    return float(grayscale.mean())


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

    train_files = set(tables["cct20_train"]["file_name"]) if "cct20_train" in tables else set()
    brightness_rows: list[dict[str, object]] = []

    train_histograms = {
        "day": np.zeros(256, dtype=np.int64),
        "night": np.zeros(256, dtype=np.int64),
    }

    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        with Image.open(input_path) as image:
            image = image.convert("RGB")
            brightness_before = compute_brightness(image)
            value_channel = extract_value_channel(image)

        brightness_rows.append(
            {
                "file_name": row.file_name,
                "split": getattr(row, "split", None),
                "day_night": getattr(row, "day_night", None),
                "brightness_before": brightness_before,
            }
        )

        if row.file_name in train_files and row.day_night in train_histograms:
            accumulate_histogram(train_histograms[row.day_night], value_channel)

    target_histogram = train_histograms["day"] + train_histograms["night"]
    lookup_tables = {
        group_name: build_lookup_table(group_hist, target_histogram)
        for group_name, group_hist in train_histograms.items()
    }

    brightness_after: dict[str, float] = {}
    transformed_count = 0
    non_train_unchanged_count = 0
    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        output_path = output_images_dir / row.file_name

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            is_train_image = row.file_name in train_files
            if is_train_image and row.day_night in lookup_tables:
                transformed = apply_histogram_lookup(image, lookup_tables[row.day_night])
                transformed_count += 1
            else:
                transformed = image
                if not is_train_image:
                    non_train_unchanged_count += 1

            save_image(transformed, output_path)
            brightness_after[row.file_name] = compute_brightness(transformed)

    brightness_df = pd.DataFrame(brightness_rows)
    brightness_df["brightness_after"] = brightness_df["file_name"].map(brightness_after)
    brightness_df["is_train_image"] = brightness_df["file_name"].isin(train_files)
    brightness_df["was_transformed"] = brightness_df["is_train_image"] & brightness_df["day_night"].isin(lookup_tables)
    brightness_df.to_csv(variant_root / "metadata" / "brightness_statistics.csv", index=False)

    histogram_df = pd.DataFrame(
        {
            "value": np.arange(256, dtype=int),
            "train_day_count": train_histograms["day"],
            "train_night_count": train_histograms["night"],
            "target_count": target_histogram,
            "day_lookup_value": lookup_tables["day"],
            "night_lookup_value": lookup_tables["night"],
        }
    )
    histogram_df.to_csv(variant_root / "metadata" / "histmatch_reference.csv", index=False)

    train_group_statistics = {}
    for group_name, group_hist in train_histograms.items():
        train_rows = brightness_df[
            brightness_df["file_name"].isin(train_files) & (brightness_df["day_night"] == group_name)
        ]
        train_group_statistics[group_name] = {
            "mean_brightness_before": float(train_rows["brightness_before"].mean()) if not train_rows.empty else None,
            "std_brightness_before": float(train_rows["brightness_before"].std(ddof=0)) if not train_rows.empty else None,
            "count_images": int(len(train_rows)),
            "count_pixels": int(group_hist.sum()),
        }

    metadata = {
        "variant_name": args.variant_name,
        "source_root": str(source_root),
        "resolved_source_images_dir": str(source_images_dir),
        "variant_root": str(variant_root),
        "num_images_processed": int(len(master)),
        "intervention": {
            "type": "histogram_matching",
            "transform_scope": "train_only",
            "evaluation_scope": "validation_and_test_images_left_original",
            "color_space": "HSV_value_channel",
            "target_mode": args.target_mode,
            "target_distribution": "combined_train_day_and_night_histogram",
            "fit_statistics_source": "training_split_only",
            "train_transformed_count": transformed_count,
            "non_train_unchanged_count": non_train_unchanged_count,
            "train_group_statistics": train_group_statistics,
        },
    }
    save_json(variant_root / "metadata" / "brightness_alignment.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
