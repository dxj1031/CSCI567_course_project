from __future__ import annotations

import argparse
import copy
import json
import os
import random
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CCT20 image classification baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--smoke", action="store_true", help="Run a short 1-epoch smoke pass.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build datasets and run a single forward pass without training.",
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
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.images_dir = images_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.images_dir / row["file_name"]
        image = Image.open(image_path).convert("RGB")
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
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_transform, eval_transform = build_transforms(img_size)
    loaders: dict[str, DataLoader] = {}

    for split_name, frame in dataframes.items():
        transform = train_transform if split_name == "train" else eval_transform
        dataset = CCT20Dataset(frame=frame, images_dir=images_dir, class_to_idx=class_to_idx, transform=transform)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders, class_to_idx


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
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


def train(config: dict[str, Any], smoke: bool, validate_only: bool) -> Path:
    set_seed(int(config.get("seed", 42)))

    images_dir = Path(config["paths"]["images_dir"])
    output_root = Path(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_run_dir(output_root, config["experiment_name"])

    train_spec, val_spec, test_specs = build_split_specs(config)
    dataframes = {
        "train": read_split(train_spec),
        "val": read_split(val_spec),
    }
    for spec in test_specs:
        dataframes[spec.name] = read_split(spec)

    explicit_class_names = config.get("label_space", {}).get("class_names")
    dataframes, class_names = apply_label_space(dataframes, explicit_class_names)
    verify_image_paths(dataframes, images_dir)

    training_cfg = copy.deepcopy(config["training"])
    if smoke:
        training_cfg["epochs"] = 1
        training_cfg["limit_train_batches"] = 2
        training_cfg["limit_eval_batches"] = 2

    loaders, class_to_idx = build_dataloaders(
        dataframes=dataframes,
        images_dir=images_dir,
        class_names=class_names,
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        img_size=int(config["model"]["img_size"]),
    )

    device = select_device(config.get("system", {}).get("device", "auto"))
    model = build_model(
        model_name=config["model"]["name"],
        num_classes=len(class_names),
        pretrained=bool(config["model"]["pretrained"]),
    ).to(device)

    forward_check = run_forward_check(model, loaders["train"], device)
    dataset_summary = {
        "images_dir": str(images_dir),
        "device": str(device),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "split_sizes": {split_name: int(len(frame)) for split_name, frame in dataframes.items()},
        "forward_check": forward_check,
        "config_path": config["_config_path"],
    }
    save_json(run_dir / "dataset_summary.json", dataset_summary)
    save_json(run_dir / "resolved_config.json", config)

    if validate_only:
        print(json.dumps(dataset_summary, indent=2))
        return run_dir

    class_weights = compute_class_weights(dataframes["train"], class_names, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_score": best_score,
        "class_names": class_names,
    }

    eval_splits = ["val"] + [spec.name for spec in test_specs]
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
    train(config=config, smoke=args.smoke, validate_only=args.validate_only)
