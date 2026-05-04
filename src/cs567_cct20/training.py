from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from .interventions import TRAIN_INTERVENTIONS, TrainImageIntervention, build_train_intervention
from .visibility import (
    VISIBILITY_MODES,
    VISIBILITY_SCOPES,
    VisibilityPreprocessor,
    build_visibility_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CCT20 image classification baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--smoke", action="store_true", help="Run a short 1-epoch smoke pass.")
    parser.add_argument(
        "--train-intervention",
        "--train_intervention",
        choices=sorted(TRAIN_INTERVENTIONS),
        default=None,
        help="Optional train-only image intervention. Overrides training.train_intervention in the YAML config.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build datasets and run a single forward pass without training.",
    )
    parser.add_argument(
        "--visibility-mode",
        choices=sorted(VISIBILITY_MODES),
        default=None,
        help="Visibility preprocessing mode. Overrides visibility.mode in the YAML config.",
    )
    parser.add_argument(
        "--visibility-scope",
        choices=sorted(VISIBILITY_SCOPES),
        default=None,
        help="Visibility preprocessing scope. Overrides visibility.scope in the YAML config.",
    )
    parser.add_argument(
        "--night-only-flag-source",
        default=None,
        help="Metadata field or rule used to identify night/low-light rows, e.g. day_night or day_night=night.",
    )
    parser.add_argument(
        "--visibility-gamma",
        type=float,
        default=None,
        help="Gamma value for gamma and gamma+CLAHE visibility modes.",
    )
    parser.add_argument(
        "--visibility-clahe-clip-limit",
        type=float,
        default=None,
        help="CLAHE clip limit for CLAHE visibility modes.",
    )
    parser.add_argument(
        "--visibility-clahe-tile-grid-size",
        default=None,
        help="CLAHE tile grid size, e.g. 8,8 or 8x8.",
    )
    parser.add_argument(
        "--eval-only-checkpoint",
        default=None,
        help="Evaluate a fixed checkpoint without retraining.",
    )
    parser.add_argument(
        "--checkpoint-results-root",
        default=None,
        help="When --eval-only-checkpoint is omitted, discover the latest matching baseline checkpoint here.",
    )
    return parser.parse_args()


def expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_value(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = expand_value(config)
    config["_config_path"] = str(config_path)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class SplitSpec:
    name: str
    csv_path: Path
    day_night: str | None


class CCT20Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: Path,
        class_to_idx: dict[str, int],
        transform: T.Compose,
        split_name: str,
        image_intervention: TrainImageIntervention | None = None,
        visibility_preprocessor: VisibilityPreprocessor | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.images_dir = images_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.split_name = split_name
        self.image_intervention = image_intervention
        self.visibility_preprocessor = visibility_preprocessor
        self.intervention_name = image_intervention.name if image_intervention is not None else "none"
        self.visibility_name = visibility_preprocessor.name if visibility_preprocessor is not None else "none"

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.images_dir / row["file_name"]
        image = Image.open(image_path).convert("RGB")
        if self.visibility_preprocessor is not None:
            image = self.visibility_preprocessor(image, row, self.split_name)
        if self.image_intervention is not None:
            image = self.image_intervention(image, row)
        image = self.transform(image)
        label = self.class_to_idx[row["category_name"]]
        return {
            "image": image,
            "label": label,
            "file_name": row["file_name"],
            "category_name": row["category_name"],
        }


def build_split_specs(config: dict[str, Any]) -> tuple[SplitSpec, SplitSpec, list[SplitSpec]]:
    train_cfg = config["datasets"]["train"]
    val_cfg = config["datasets"]["val"]
    test_cfg = config["datasets"].get("tests", {})

    train_spec = SplitSpec(
        name="train",
        csv_path=Path(train_cfg["csv"]),
        day_night=train_cfg.get("filters", {}).get("day_night"),
    )
    val_spec = SplitSpec(
        name="val",
        csv_path=Path(val_cfg["csv"]),
        day_night=val_cfg.get("filters", {}).get("day_night"),
    )
    test_specs = [
        SplitSpec(
            name=name,
            csv_path=Path(spec["csv"]),
            day_night=spec.get("filters", {}).get("day_night"),
        )
        for name, spec in test_cfg.items()
    ]
    return train_spec, val_spec, test_specs


def read_split(spec: SplitSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.csv_path)
    if spec.day_night is not None:
        frame = frame[frame["day_night"] == spec.day_night].copy()
    return frame.reset_index(drop=True)


def resolve_images_dir(images_dir: Path, dataframes: dict[str, pd.DataFrame]) -> Path:
    candidates = [images_dir]
    if images_dir.exists():
        candidates.extend(sorted(path for path in images_dir.iterdir() if path.is_dir()))

    sample_files: list[str] = []
    for frame in dataframes.values():
        if frame.empty:
            continue
        sample_files.extend(frame["file_name"].head(10).tolist())
        if len(sample_files) >= 20:
            break

    sample_files = list(dict.fromkeys(sample_files))
    if not sample_files:
        raise ValueError("No image file names were available to resolve the image directory.")

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


def apply_label_space(
    dataframes: dict[str, pd.DataFrame],
    explicit_class_names: list[str] | None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    train_frame = dataframes["train"]
    if explicit_class_names:
        class_names = explicit_class_names
    else:
        class_names = sorted(train_frame["category_name"].unique().tolist())

    filtered: dict[str, pd.DataFrame] = {}
    for split_name, frame in dataframes.items():
        split_frame = frame[frame["category_name"].isin(class_names)].copy()
        if split_frame.empty:
            raise ValueError(f"Split '{split_name}' is empty after applying the label space.")
        filtered[split_name] = split_frame.reset_index(drop=True)
    return filtered, class_names


def verify_image_paths(dataframes: dict[str, pd.DataFrame], images_dir: Path) -> None:
    missing: list[str] = []
    seen: set[str] = set()
    for frame in dataframes.values():
        for file_name in frame["file_name"].tolist():
            if file_name in seen:
                continue
            seen.add(file_name)
            if not (images_dir / file_name).exists():
                missing.append(file_name)
            if len(missing) >= 10:
                break
        if missing:
            break
    if missing:
        raise FileNotFoundError(f"Missing image files under {images_dir}: {missing}")


def build_transforms(img_size: int) -> tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            normalize,
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    dataframes: dict[str, pd.DataFrame],
    images_dir: Path,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    img_size: int,
    train_intervention: TrainImageIntervention,
    visibility_preprocessor: VisibilityPreprocessor | None,
    seed: int,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_transform, eval_transform = build_transforms(img_size)
    loaders: dict[str, DataLoader] = {}
    generator = torch.Generator()
    generator.manual_seed(seed)

    for split_name, frame in dataframes.items():
        transform = train_transform if split_name == "train" else eval_transform
        image_intervention = train_intervention if split_name == "train" and train_intervention.name != "none" else None
        dataset = CCT20Dataset(
            frame=frame,
            images_dir=images_dir,
            class_to_idx=class_to_idx,
            transform=transform,
            split_name=split_name,
            image_intervention=image_intervention,
            visibility_preprocessor=visibility_preprocessor,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator if split_name == "train" else None,
        )
    return loaders, class_to_idx


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif model_name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def compute_class_weights(train_frame: pd.DataFrame, class_names: list[str], device: torch.device) -> torch.Tensor:
    counts = train_frame["category_name"].value_counts()
    weights = []
    total = float(len(train_frame))
    num_classes = float(len(class_names))
    for class_name in class_names:
        class_count = float(counts[class_name])
        weights.append(total / (num_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_forward_check(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(images)
    return {
        "batch_size": int(images.shape[0]),
        "num_classes": int(logits.shape[1]),
        "image_shape": [int(dim) for dim in images.shape],
        "logit_shape": [int(dim) for dim in logits.shape],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    limit_batches: int | None,
) -> float:
    model.train()
    running_loss = 0.0
    batches_seen = 0

    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches_seen += 1

    if batches_seen == 0:
        raise RuntimeError("No training batches were processed.")
    return running_loss / batches_seen


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    limit_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches_seen = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_files: list[str] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            total_loss += loss.item()
            batches_seen += 1
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_files.extend(batch["file_name"])

    if batches_seen == 0:
        raise RuntimeError("No evaluation batches were processed.")

    confusion = confusion_matrix(all_targets, all_predictions, labels=list(range(len(class_names))))
    per_class_scores = f1_score(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )

    return {
        "loss": total_loss / batches_seen,
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, average="macro", zero_division=0),
        "targets": all_targets,
        "predictions": all_predictions,
        "files": all_files,
        "confusion_matrix": confusion.tolist(),
        "per_class_f1": {class_name: float(score) for class_name, score in zip(class_names, per_class_scores)},
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def save_predictions(split_name: str, metrics: dict[str, Any], class_names: list[str], path: Path) -> None:
    rows = []
    for file_name, target_idx, prediction_idx in zip(metrics["files"], metrics["targets"], metrics["predictions"]):
        rows.append(
            {
                "file_name": file_name,
                "target_idx": target_idx,
                "target_name": class_names[target_idx],
                "prediction_idx": prediction_idx,
                "prediction_name": class_names[prediction_idx],
                "split_name": split_name,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_confusion_outputs(split_name: str, metrics: dict[str, Any], class_names: list[str], run_dir: Path) -> None:
    confusion = np.asarray(metrics["confusion_matrix"])
    confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    confusion_df.to_csv(run_dir / f"confusion_matrix_{split_name}.csv")

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_df, cmap="Blues", annot=False, square=True)
    plt.title(f"Confusion Matrix: {split_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(run_dir / f"confusion_matrix_{split_name}.png", dpi=200)
    plt.close()

    per_class_path = run_dir / f"per_class_f1_{split_name}.csv"
    pd.DataFrame(
        [{"class_name": class_name, "f1": score} for class_name, score in metrics["per_class_f1"].items()]
    ).to_csv(per_class_path, index=False)


def resolve_run_dir(output_root: Path, experiment_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def infer_experiment_fields(experiment_name: str, train_intervention_name: str) -> dict[str, str]:
    scenario = experiment_name
    backbone = "unknown"
    for model_name in ["resnet101", "resnet50", "resnet34", "resnet18"]:
        token = f"_{model_name}"
        if token in experiment_name:
            scenario = experiment_name.split(token)[0]
            backbone = model_name
            break
    variant = "original" if train_intervention_name == "none" else train_intervention_name
    return {
        "scenario": scenario,
        "backbone": backbone,
        "variant": variant,
    }


def build_visibility_overrides(args: argparse.Namespace) -> dict[str, Any]:
    cli_pairs = {
        "mode": args.visibility_mode,
        "scope": args.visibility_scope,
        "night_only_flag_source": args.night_only_flag_source,
        "gamma": args.visibility_gamma,
        "clahe_clip_limit": args.visibility_clahe_clip_limit,
        "clahe_tile_grid_size": args.visibility_clahe_tile_grid_size,
    }
    overrides = {key: value for key, value in cli_pairs.items() if value is not None}
    if overrides:
        overrides["experiment_requested"] = True
    return overrides


def append_visibility_suffix(experiment_name: str, visibility_summary: dict[str, Any], requested: bool) -> str:
    if not requested:
        return experiment_name
    suffix = f"_visibility_{visibility_summary['scope']}_{visibility_summary['mode']}"
    if experiment_name.endswith(suffix):
        return experiment_name
    return f"{experiment_name}{suffix}"


def build_visibility_sanity_checks(visibility_summary: dict[str, Any]) -> dict[str, bool]:
    counts = visibility_summary.get("application_counts", {})
    enabled = bool(visibility_summary.get("enabled"))
    scope = str(visibility_summary.get("scope"))

    all_counts = list(counts.values())
    train_count = counts.get("train", {})
    no_images_enhanced = all(int(split_count.get("enhanced_rows", 0)) == 0 for split_count in all_counts)
    enhanced_subset_of_night = all(
        int(split_count.get("enhanced_rows", 0)) <= int(split_count.get("night_rows", 0))
        for split_count in all_counts
    )
    test_only_all_eval_night_rows_enhanced = all(
        int(split_count.get("enhanced_rows", 0))
        == (0 if split_name == "train" else int(split_count.get("night_rows", 0)))
        for split_name, split_count in counts.items()
    )
    night_only_all_night_rows_enhanced = all(
        int(split_count.get("enhanced_rows", 0)) == int(split_count.get("night_rows", 0))
        for split_count in all_counts
    )
    all_rows_enhanced = all(
        int(split_count.get("enhanced_rows", 0)) == int(split_count.get("rows", 0))
        for split_count in all_counts
    )

    return {
        "visibility_original_mode_no_images_modified": (not enabled and no_images_enhanced) or enabled,
        "visibility_test_only_training_images_unmodified": (
            scope != "test_only" or int(train_count.get("enhanced_rows", 0)) == 0
        ),
        "visibility_test_only_low_light_eval_only": (
            scope != "test_only"
            or (
                int(train_count.get("enhanced_rows", 0)) == 0
                and enhanced_subset_of_night
                and ((not enabled) or test_only_all_eval_night_rows_enhanced)
            )
        ),
        "visibility_train_test_consistent_all_splits_same_policy": (
            scope != "train_test_consistent" or (not enabled) or all_rows_enhanced
        ),
        "visibility_night_only_only_low_light_rows": (
            scope != "night_only"
            or (
                enhanced_subset_of_night
                and ((not enabled) or night_only_all_night_rows_enhanced)
            )
        ),
    }


def resolve_eval_checkpoint(
    explicit_checkpoint: str | None,
    checkpoint_results_root: str | None,
    baseline_experiment_name: str,
) -> Path | None:
    if explicit_checkpoint:
        return Path(os.path.expandvars(os.path.expanduser(explicit_checkpoint))).resolve()
    if not checkpoint_results_root:
        return None

    results_root = Path(os.path.expandvars(os.path.expanduser(checkpoint_results_root))).resolve()
    candidates: list[tuple[float, str, Path]] = []
    for summary_path in results_root.rglob("summary.json"):
        try:
            summary = load_json(summary_path)
        except (json.JSONDecodeError, OSError):
            continue
        if summary.get("experiment_name") != baseline_experiment_name:
            continue

        checkpoint_value = summary.get("checkpoint")
        checkpoint_path = Path(checkpoint_value) if checkpoint_value else summary_path.parent / "best.pt"
        if not checkpoint_path.is_absolute():
            checkpoint_path = summary_path.parent / checkpoint_path
        if not checkpoint_path.exists():
            fallback_path = summary_path.parent / "best.pt"
            if fallback_path.exists():
                checkpoint_path = fallback_path
        if checkpoint_path.exists():
            candidates.append((summary_path.stat().st_mtime, summary_path.parent.name, checkpoint_path))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a checkpoint for experiment_name={baseline_experiment_name!r} under {results_root}."
        )

    return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][2]


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected checkpoint {path} to contain a dict payload.")
    return payload


def train(
    config: dict[str, Any],
    smoke: bool,
    validate_only: bool,
    train_intervention_override: str | None = None,
    visibility_overrides: dict[str, Any] | None = None,
    eval_only_checkpoint: str | None = None,
    checkpoint_results_root: str | None = None,
) -> Path:
    set_seed(int(config.get("seed", 42)))

    baseline_experiment_name = str(config["experiment_name"])
    has_visibility_config = "visibility" in config
    configured_images_dir = Path(config["paths"]["images_dir"])
    output_root = Path(config["paths"]["output_root"])
    training_cfg = copy.deepcopy(config["training"])
    if train_intervention_override is not None:
        training_cfg["train_intervention"] = train_intervention_override
        config["training"]["train_intervention"] = train_intervention_override
    if smoke:
        training_cfg["epochs"] = 1
        training_cfg["limit_train_batches"] = 2
        training_cfg["limit_eval_batches"] = 2

    train_intervention_name = str(training_cfg.get("train_intervention", "none")).lower()
    if train_intervention_name != "none" and not config["experiment_name"].endswith(f"_{train_intervention_name}"):
        config["experiment_name"] = f"{config['experiment_name']}_{train_intervention_name}"

    visibility_settings = build_visibility_settings(config, visibility_overrides)
    visibility_requested = bool(visibility_settings.experiment_requested or has_visibility_config)
    visibility_summary_for_config = visibility_settings.to_dict()
    config["visibility"] = visibility_summary_for_config
    config["experiment_name"] = append_visibility_suffix(
        str(config["experiment_name"]),
        visibility_summary_for_config,
        requested=visibility_requested,
    )

    eval_only_requested = bool(eval_only_checkpoint or checkpoint_results_root)
    if eval_only_requested and train_intervention_name != "none":
        raise ValueError("Evaluation-only visibility runs must keep training.train_intervention set to none.")
    checkpoint_path_for_eval = None
    if eval_only_requested and not validate_only:
        checkpoint_path_for_eval = resolve_eval_checkpoint(
            explicit_checkpoint=eval_only_checkpoint,
            checkpoint_results_root=checkpoint_results_root,
            baseline_experiment_name=baseline_experiment_name,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_run_dir(output_root, config["experiment_name"])
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = run_dir.name

    train_spec, val_spec, test_specs = build_split_specs(config)
    dataframes = {
        "train": read_split(train_spec),
        "val": read_split(val_spec),
    }
    for spec in test_specs:
        dataframes[spec.name] = read_split(spec)

    explicit_class_names = config.get("label_space", {}).get("class_names")
    dataframes, class_names = apply_label_space(dataframes, explicit_class_names)
    images_dir = resolve_images_dir(configured_images_dir, dataframes)
    verify_image_paths(dataframes, images_dir)

    train_intervention = build_train_intervention(
        name=train_intervention_name,
        train_frame=dataframes["train"],
        images_dir=images_dir,
        dataset_root=configured_images_dir.parent,
        params=training_cfg.get("train_intervention_params", {}),
    )
    visibility_preprocessor = VisibilityPreprocessor(visibility_settings)
    visibility_summary = visibility_preprocessor.summary(dataframes)
    dataset_visibility_preprocessor = (
        visibility_preprocessor if visibility_requested or visibility_settings.enabled else None
    )

    loaders, class_to_idx = build_dataloaders(
        dataframes=dataframes,
        images_dir=images_dir,
        class_names=class_names,
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        img_size=int(config["model"]["img_size"]),
        train_intervention=train_intervention,
        visibility_preprocessor=dataset_visibility_preprocessor,
        seed=int(config.get("seed", 42)),
    )

    device = select_device(config.get("system", {}).get("device", "auto"))
    model = build_model(
        model_name=config["model"]["name"],
        num_classes=len(class_names),
        pretrained=bool(config["model"]["pretrained"]),
    ).to(device)

    forward_check = run_forward_check(model, loaders["train"], device)
    split_interventions = {
        split_name: getattr(loader.dataset, "intervention_name", "none")
        for split_name, loader in loaders.items()
    }
    split_visibility_preprocessing = {
        split_name: getattr(loader.dataset, "visibility_name", "none")
        for split_name, loader in loaders.items()
    }
    sanity_checks = {
        "train_intervention_only": all(
            intervention_name == "none"
            for split_name, intervention_name in split_interventions.items()
            if split_name != "train"
        ),
        "validation_and_test_images_unmodified_by_intervention": True,
        "test_domain_statistics_used": False,
        **build_visibility_sanity_checks(visibility_summary),
    }
    dataset_summary = {
        "run_id": run_id,
        "timestamp": run_timestamp,
        "configured_images_dir": str(configured_images_dir),
        "resolved_images_dir": str(images_dir),
        "device": str(device),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "split_sizes": {split_name: int(len(frame)) for split_name, frame in dataframes.items()},
        "train_intervention": train_intervention.summary(),
        "split_interventions": split_interventions,
        "visibility_preprocessing": visibility_summary,
        "split_visibility_preprocessing": split_visibility_preprocessing,
        "eval_only": eval_only_requested,
        "source_checkpoint": str(checkpoint_path_for_eval) if checkpoint_path_for_eval is not None else None,
        "sanity_checks": sanity_checks,
        "forward_check": forward_check,
        "config_path": config["_config_path"],
    }
    save_json(run_dir / "dataset_summary.json", dataset_summary)
    save_json(run_dir / "resolved_config.json", config)
    save_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "timestamp": run_timestamp,
            "experiment_name": config["experiment_name"],
            **infer_experiment_fields(config["experiment_name"], train_intervention_name),
            "seed": int(config.get("seed", 42)),
            "config_path": config["_config_path"],
            "command": " ".join([sys.executable, *sys.argv]),
            "argv": [sys.executable, *sys.argv],
            "data_split": {
                "train": str(train_spec.csv_path),
                "val": str(val_spec.csv_path),
                "tests": {spec.name: str(spec.csv_path) for spec in test_specs},
            },
            "train_intervention": train_intervention.summary(),
            "visibility_preprocessing": visibility_summary,
            "training_settings": training_cfg,
            "eval_only": eval_only_requested,
            "source_checkpoint": str(checkpoint_path_for_eval) if checkpoint_path_for_eval is not None else None,
            "output_files": {
                "dataset_summary": str(run_dir / "dataset_summary.json"),
                "resolved_config": str(run_dir / "resolved_config.json"),
                "run_manifest": str(run_dir / "run_manifest.json"),
                "summary": str(run_dir / "summary.json"),
            },
            "notes": (
                "Visibility hypothesis run. If eval_only is true, the checkpoint is fixed and no retraining is done; "
                "otherwise the logged visibility preprocessing is part of the training and evaluation data pipeline."
            ),
        },
    )

    if validate_only:
        print(json.dumps(dataset_summary, indent=2))
        return run_dir

    class_weights = compute_class_weights(dataframes["train"], class_names, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    eval_splits = ["val"] + [spec.name for spec in test_specs]
    if eval_only_requested:
        if checkpoint_path_for_eval is None:
            raise ValueError("Evaluation-only mode requires --eval-only-checkpoint or --checkpoint-results-root.")
        checkpoint_payload = load_checkpoint(checkpoint_path_for_eval, device)
        checkpoint_class_names = checkpoint_payload.get("class_names")
        if checkpoint_class_names is not None and list(checkpoint_class_names) != class_names:
            raise ValueError(
                "Checkpoint class_names do not match the active config label_space: "
                f"{checkpoint_class_names!r} != {class_names!r}"
            )
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        model.load_state_dict(state_dict)

        summary: dict[str, Any] = {
            "experiment_name": config["experiment_name"],
            "run_id": run_id,
            "timestamp": run_timestamp,
            "seed": int(config.get("seed", 42)),
            "checkpoint": str(checkpoint_path_for_eval),
            "source_experiment_name": baseline_experiment_name,
            "eval_only": True,
            "best_epoch": checkpoint_payload.get("epoch"),
            "selection_metric": training_cfg.get("selection_metric", "macro_f1"),
            "best_score": checkpoint_payload.get("score"),
            "class_names": class_names,
            "train_intervention": train_intervention.summary(),
            "visibility_preprocessing": visibility_summary,
        }

        for split_name in eval_splits:
            metrics = evaluate(
                model=model,
                loader=loaders[split_name],
                criterion=criterion,
                device=device,
                class_names=class_names,
                limit_batches=training_cfg.get("limit_eval_batches"),
            )
            reduced_metrics = {
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": metrics["per_class_f1"],
            }
            save_json(run_dir / f"metrics_{split_name}.json", reduced_metrics)
            save_predictions(split_name, metrics, class_names, run_dir / f"predictions_{split_name}.csv")
            save_confusion_outputs(split_name, metrics, class_names, run_dir)
            summary[split_name] = reduced_metrics

        save_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return run_dir

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    patience = int(training_cfg["early_stopping_patience"])
    stale_epochs = 0
    selection_metric = training_cfg.get("selection_metric", "macro_f1")

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            limit_batches=training_cfg.get("limit_train_batches"),
        )
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            class_names=class_names,
            limit_batches=training_cfg.get("limit_eval_batches"),
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row))

        current_score = float(val_metrics[selection_metric])
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "score": best_score,
                "class_names": class_names,
                "config": config,
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training ended without producing a checkpoint.")

    checkpoint_path = run_dir / "best.pt"
    torch.save(best_state, checkpoint_path)
    save_history(history, run_dir / "history.csv")

    model.load_state_dict(best_state["model_state_dict"])

    summary: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "run_id": run_id,
        "timestamp": run_timestamp,
        "seed": int(config.get("seed", 42)),
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_score": best_score,
        "class_names": class_names,
        "train_intervention": train_intervention.summary(),
        "visibility_preprocessing": visibility_summary,
        "eval_only": False,
    }

    for split_name in eval_splits:
        metrics = evaluate(
            model=model,
            loader=loaders[split_name],
            criterion=criterion,
            device=device,
            class_names=class_names,
            limit_batches=training_cfg.get("limit_eval_batches"),
        )
        reduced_metrics = {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": metrics["per_class_f1"],
        }
        save_json(run_dir / f"metrics_{split_name}.json", reduced_metrics)
        save_predictions(split_name, metrics, class_names, run_dir / f"predictions_{split_name}.csv")
        save_confusion_outputs(split_name, metrics, class_names, run_dir)
        summary[split_name] = reduced_metrics

    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(
        config=config,
        smoke=args.smoke,
        validate_only=args.validate_only,
        train_intervention_override=args.train_intervention,
        visibility_overrides=build_visibility_overrides(args),
        eval_only_checkpoint=args.eval_only_checkpoint,
        checkpoint_results_root=args.checkpoint_results_root,
    )
