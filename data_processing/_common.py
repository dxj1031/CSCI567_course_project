from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


CSV_FILE_NAMES = [
    "cct20_clean_all.csv",
    "cct20_train.csv",
    "cct20_val.csv",
    "cct20_cis.csv",
    "cct20_trans.csv",
    "cct20_train_day.csv",
    "cct20_train_night.csv",
]

METADATA_FILE_NAMES = [
    "label_mapping.json",
    "data_processing_spec.json",
]


def load_processed_tables(source_root: Path) -> dict[str, pd.DataFrame]:
    processed_dir = source_root / "processed"
    tables: dict[str, pd.DataFrame] = {}
    for file_name in CSV_FILE_NAMES:
        path = processed_dir / file_name
        if path.exists():
            tables[path.stem] = pd.read_csv(path)
    return tables


def resolve_source_images_dir(source_root: Path, sample_files: list[str]) -> Path:
    images_dir = source_root / "images"
    candidates = [images_dir]
    if images_dir.exists():
        candidates.extend(sorted(path for path in images_dir.iterdir() if path.is_dir()))

    best_candidate = images_dir
    best_match_count = -1

    for candidate in candidates:
        match_count = sum((candidate / file_name).exists() for file_name in sample_files)
        if match_count > best_match_count:
            best_candidate = candidate
            best_match_count = match_count
        if match_count == len(sample_files):
            return candidate

    checked = [str(path) for path in candidates]
    raise FileNotFoundError(
        f"Could not resolve an image root under {images_dir}. "
        f"Checked candidates: {checked}. Best match count: {best_match_count}/{len(sample_files)}"
    )


def copy_processed_metadata(source_root: Path, variant_root: Path) -> None:
    source_processed_dir = source_root / "processed"
    target_processed_dir = variant_root / "processed"
    target_processed_dir.mkdir(parents=True, exist_ok=True)

    for file_name in CSV_FILE_NAMES + METADATA_FILE_NAMES:
        source_path = source_processed_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, target_processed_dir / file_name)


def build_variant_root(output_root: Path, variant_name: str) -> Path:
    variant_root = output_root / variant_name
    (variant_root / "images").mkdir(parents=True, exist_ok=True)
    (variant_root / "metadata").mkdir(parents=True, exist_ok=True)
    return variant_root


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_master_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "cct20_clean_all" not in tables:
        raise KeyError("Expected cct20_clean_all.csv to exist in the processed directory.")

    master = tables["cct20_clean_all"].copy()
    subset_columns = [column for column in ["file_name", "split", "day_night", "category_name"] if column in master.columns]
    return master[subset_columns].drop_duplicates("file_name").reset_index(drop=True)
