#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from _common import (
    build_master_table,
    build_variant_root,
    copy_processed_metadata,
    load_processed_tables,
    resolve_source_images_dir,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a background-blurred CCT20 dataset variant.")
    parser.add_argument("--source-root", required=True, help="Original dataset root containing images/ and processed/.")
    parser.add_argument("--output-root", required=True, help="Directory where dataset variants should be stored.")
    parser.add_argument("--variant-name", default="dataset_bg_blur", help="Name of the output dataset variant.")
    parser.add_argument("--blur-radius", type=float, default=8.0, help="Gaussian blur radius for the background.")
    parser.add_argument(
        "--center-scale",
        type=float,
        default=0.55,
        help="Relative width/height of the preserved central foreground region.",
    )
    parser.add_argument(
        "--mask-feather",
        type=float,
        default=18.0,
        help="Gaussian blur radius applied to the foreground mask for smooth transitions.",
    )
    return parser.parse_args()


def apply_background_blur(
    image: Image.Image,
    blur_radius: float,
    center_scale: float,
    mask_feather: float,
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    center_width = width * center_scale
    center_height = height * center_scale

    left = (width - center_width) / 2
    top = (height - center_height) / 2
    right = left + center_width
    bottom = top + center_height

    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((left, top, right, bottom), fill=255)
    if mask_feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_feather))

    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.composite(image, blurred, mask)


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
    processed_count = 0

    for row in master.itertuples(index=False):
        input_path = source_images_dir / row.file_name
        output_path = output_images_dir / row.file_name

        with Image.open(input_path) as image:
            transformed = apply_background_blur(
                image=image,
                blur_radius=args.blur_radius,
                center_scale=args.center_scale,
                mask_feather=args.mask_feather,
            )
            save_image(transformed, output_path)
        processed_count += 1

    metadata = {
        "variant_name": args.variant_name,
        "source_root": str(source_root),
        "resolved_source_images_dir": str(source_images_dir),
        "variant_root": str(variant_root),
        "num_images_processed": processed_count,
        "intervention": {
            "type": "background_blur",
            "blur_radius": args.blur_radius,
            "center_scale": args.center_scale,
            "mask_feather": args.mask_feather,
        },
    }
    save_json(variant_root / "metadata" / "background_intervention.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
