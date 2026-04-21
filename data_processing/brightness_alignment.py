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
    parser = argparse.ArgumentParser(description="Create a brightness-aligned CCT20 dataset variant.")
    parser.add_argument("--source-root", required=True, help="Original dataset root containing images/ and processed/.")
    parser.add_argument("--output-root", required=True, help="Directory where dataset variants should be stored.")
    parser.add_argument(
        "--variant-name",
        default="dataset_brightness_aligned",
        help="Name of the output dataset variant.",
    )
    return parser.parse_args()


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_brightness(image: Image.Image) -> float:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    return float(grayscale.mean())


def apply_brightness_alignment(
    image: Image.Image,
    source_mean: float,
    source_std: float,
    target_mean: float,
    target_std: float,
) -> Image.Image:
    rgb = image_to_rgb_array(image)
    image_float = Image.fromarray(rgb.astype(np.uint8), mode="RGB").convert("HSV")
    hsv = np.asarray(image_float, dtype=np.float32)

    value = hsv[:, :, 2]
    safe_std = source_std if source_std > 1e-6 else 1.0
    aligned_value = ((value - source_mean) / safe_std) * target_std + target_mean
    hsv[:, :, 2] = np.clip(aligned_value, 0.0, 255.0)

    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def build_group_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for day_night, subset in frame.groupby("day_night"):
        stats[day_night] = {
            "mean": float(subset["brightness_before"].mean()),
            "std": float(subset["brightness_before"].std(ddof=0)),
            "count": int(len(subset)),
        }
    return stats


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
    train_frame = tables["cct20_train"].copy()
    source_images_dir = resolve_source_images_dir(source_root, master["file_name"].head(32).tolist())
    variant_root = build_variant_root(output_root, args.variant_name)
    copy_processed_metadata(source_root, variant_root)
    output_images_dir = variant_root / "images"

    brightness_rows: list[dict[str, object]] = []
    raw_brightness: dict[str, float] = {}

    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        with Image.open(input_path) as image:
            brightness = compute_brightness(image)
        raw_brightness[row.file_name] = brightness
        brightness_rows.append(
            {
                "file_name": row.file_name,
                "split": getattr(row, "split", None),
                "day_night": getattr(row, "day_night", None),
                "brightness_before": brightness,
            }
        )

    brightness_df = pd.DataFrame(brightness_rows)
    train_brightness = brightness_df[brightness_df["file_name"].isin(train_frame["file_name"])]
    train_brightness = train_brightness[train_brightness["day_night"].isin(["day", "night"])].copy()

    target_mean = float(train_brightness["brightness_before"].mean())
    target_std = float(train_brightness["brightness_before"].std(ddof=0))
    group_stats = build_group_statistics(train_brightness)

    brightness_after: dict[str, float] = {}
    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        output_path = output_images_dir / row.file_name

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            if row.day_night in group_stats:
                stats = group_stats[row.day_night]
                transformed = apply_brightness_alignment(
                    image=image,
                    source_mean=stats["mean"],
                    source_std=stats["std"],
                    target_mean=target_mean,
                    target_std=target_std if target_std > 1e-6 else 1.0,
                )
            else:
                transformed = image

            save_image(transformed, output_path)
            brightness_after[row.file_name] = compute_brightness(transformed)

    brightness_df["brightness_after"] = brightness_df["file_name"].map(brightness_after)
    brightness_df.to_csv(variant_root / "metadata" / "brightness_statistics.csv", index=False)

    metadata = {
        "variant_name": args.variant_name,
        "source_root": str(source_root),
        "resolved_source_images_dir": str(source_images_dir),
        "variant_root": str(variant_root),
        "intervention": {
            "type": "brightness_alignment",
            "method": "mean_std_alignment_on_hsv_value",
            "target_mean": target_mean,
            "target_std": target_std,
            "train_group_statistics": group_stats,
        },
        "num_images_processed": int(len(master)),
    }
    save_json(variant_root / "metadata" / "brightness_alignment.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
