#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from compare_capacity import classify_split_domain, dataframe_to_markdown, find_summary_files, load_json


SCENARIO_PATTERN = re.compile(r"(?P<scenario>cross_location|day_to_night|night_to_day)_resnet(?P<depth>18|34|50|101)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate seed, class-balance, and object-centric follow-up runs.")
    parser.add_argument("--results-root", required=True, help="Directory containing run folders with summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory where follow-up tables should be written.")
    return parser.parse_args()


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def infer_scenario_backbone(experiment_name: str) -> tuple[str, str, int | None]:
    match = SCENARIO_PATTERN.search(experiment_name.lower())
    if match is None:
        return "unknown", "unknown", None
    depth = int(match.group("depth"))
    return match.group("scenario"), f"resnet{depth}", depth


def clean_group_value(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def summarize_generalization(split_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not split_rows:
        return pd.DataFrame()
    split_df = pd.DataFrame(split_rows)
    rows: list[dict[str, Any]] = []
    group_columns = [
        "experiment_name",
        "run_id",
        "seed",
        "scenario",
        "backbone",
        "depth",
        "train_intervention",
        "loss",
        "class_weight_mode",
        "train_sampler",
        "image_ablation",
    ]
    for group_key, frame in split_df.groupby(group_columns, dropna=False, sort=True):
        payload = dict(zip(group_columns, group_key))
        classified = frame.copy()
        classified["domain_role"] = classified["split"].map(
            lambda split_name: classify_split_domain(str(split_name), str(payload["scenario"]))
        )
        in_domain = classified[classified["domain_role"] == "in"]
        out_domain = classified[classified["domain_role"] == "out"]
        if in_domain.empty or out_domain.empty:
            continue
        in_acc = float(in_domain["accuracy"].mean())
        ood_acc = float(out_domain["accuracy"].mean())
        gap = in_acc - ood_acc
        payload.update(
            {
                "in_domain_accuracy": in_acc,
                "ood_accuracy": ood_acc,
                "gap": gap,
                "normalized_gap": gap / in_acc if in_acc else None,
                "in_domain_macro_f1": float(in_domain["macro_f1"].mean()),
                "ood_macro_f1": float(out_domain["macro_f1"].mean()),
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize_by_group(generalization_df: pd.DataFrame) -> pd.DataFrame:
    if generalization_df.empty:
        return pd.DataFrame()
    group_columns = [
        "scenario",
        "backbone",
        "train_intervention",
        "loss",
        "class_weight_mode",
        "train_sampler",
        "image_ablation",
    ]
    rows: list[dict[str, Any]] = []
    for group_key, frame in generalization_df.groupby(group_columns, dropna=False, sort=True):
        payload = dict(zip(group_columns, group_key))
        payload.update(
            {
                "runs": int(len(frame)),
                "seeds": ",".join(str(seed) for seed in sorted(frame["seed"].dropna().unique())),
                "mean_ood_accuracy": float(frame["ood_accuracy"].mean()),
                "std_ood_accuracy": float(frame["ood_accuracy"].std(ddof=0)),
                "mean_normalized_gap": float(frame["normalized_gap"].mean()),
                "std_normalized_gap": float(frame["normalized_gap"].std(ddof=0)),
                "mean_ood_macro_f1": float(frame["ood_macro_f1"].mean()),
                "std_ood_macro_f1": float(frame["ood_macro_f1"].std(ddof=0)),
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_rows(summary_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        experiment_name = str(summary.get("experiment_name", summary_path.parent.name))
        scenario, backbone, depth = infer_scenario_backbone(experiment_name)
        if scenario == "unknown":
            continue

        run_dir = summary_path.parent
        manifest = load_optional_json(run_dir / "run_manifest.json")
        dataset_summary = load_optional_json(run_dir / "dataset_summary.json")
        training_diagnostics = summary.get("training_diagnostics") or dataset_summary.get("training_diagnostics", {})
        image_ablation_summary = summary.get("image_ablation") or dataset_summary.get("image_ablation", {})
        train_intervention_summary = summary.get("train_intervention") or dataset_summary.get("train_intervention", {})

        seed = summary.get("seed", manifest.get("seed"))
        run_id = summary.get("run_id", manifest.get("run_id", run_dir.name))
        train_intervention = clean_group_value(train_intervention_summary.get("name"), "none")
        image_ablation = clean_group_value(image_ablation_summary.get("name"), "none")
        loss = clean_group_value(training_diagnostics.get("loss"), "cross_entropy")
        class_weight_mode = clean_group_value(training_diagnostics.get("class_weight_mode"), "balanced")
        train_sampler = clean_group_value(training_diagnostics.get("train_sampler"), "none")

        base_row = {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "seed": seed,
            "scenario": scenario,
            "backbone": backbone,
            "depth": depth,
            "train_intervention": train_intervention,
            "loss": loss,
            "class_weight_mode": class_weight_mode,
            "train_sampler": train_sampler,
            "image_ablation": image_ablation,
            "run_dir": str(run_dir),
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": summary.get("selection_metric"),
            "best_score": summary.get("best_score"),
        }
        run_rows.append(base_row)

        for split_name, metrics in summary.items():
            if not isinstance(metrics, dict) or "accuracy" not in metrics or "macro_f1" not in metrics:
                continue
            split_rows.append(
                {
                    **base_row,
                    "split": split_name,
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                }
            )
            for class_name, f1_value in metrics.get("per_class_f1", {}).items():
                per_class_rows.append(
                    {
                        **base_row,
                        "split": split_name,
                        "class_name": class_name,
                        "f1": float(f1_value),
                    }
                )

    return pd.DataFrame(run_rows), pd.DataFrame(split_rows), pd.DataFrame(per_class_rows)


def write_markdown(output_dir: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Follow-Up Experiment Comparison",
        "",
        "This report aggregates the new robustness, rare-class, and object-centric diagnostic runs.",
        "",
    ]
    if summary_df.empty:
        lines.append("No matching follow-up runs were found.")
    else:
        columns = [
            "scenario",
            "backbone",
            "train_intervention",
            "loss",
            "class_weight_mode",
            "train_sampler",
            "image_ablation",
            "runs",
            "mean_ood_accuracy",
            "std_ood_accuracy",
            "mean_normalized_gap",
            "std_normalized_gap",
            "mean_ood_macro_f1",
        ]
        lines.append(dataframe_to_markdown(summary_df[columns]))
    (output_dir / "followup_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df, split_df, per_class_df = build_rows(find_summary_files(results_root))
    generalization_df = summarize_generalization(split_df.to_dict("records") if not split_df.empty else [])
    seed_summary_df = summarize_by_group(generalization_df)

    run_df.to_csv(output_dir / "followup_runs.csv", index=False)
    split_df.to_csv(output_dir / "followup_split_metrics.csv", index=False)
    per_class_df.to_csv(output_dir / "followup_per_class_f1.csv", index=False)
    generalization_df.to_csv(output_dir / "followup_generalization_metrics.csv", index=False)
    seed_summary_df.to_csv(output_dir / "followup_seed_summary.csv", index=False)
    write_markdown(output_dir, seed_summary_df)


if __name__ == "__main__":
    main()
