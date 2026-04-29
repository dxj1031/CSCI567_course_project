#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from compare_capacity import (
    classify_split_domain,
    dataframe_to_markdown,
    find_summary_files,
    load_json,
    select_preferred_summary_paths,
)


VARIANT_LABELS = {
    "original": "Original",
    "bbox_bg": "BBox Background",
    "histmatch": "Histogram Match",
}

VARIANT_ORDER = ["original", "bbox_bg", "histmatch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ResNet50 dataset-intervention experiment summaries."
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Directory containing experiment run folders with summary.json outputs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison CSV/Markdown files will be written.",
    )
    return parser.parse_args()


def infer_variant(experiment_name: str) -> str | None:
    name = experiment_name.lower()
    if not name.startswith(("cross_location_resnet50", "day_to_night_resnet50", "night_to_day_resnet50")):
        return None
    if name.endswith("_bbox_bg"):
        return "bbox_bg"
    if name.endswith("_histmatch"):
        return "histmatch"
    if re.search(r"_resnet50$", name):
        return "original"
    return None


def infer_scenario(experiment_name: str) -> str | None:
    match = re.match(
        r"(?P<scenario>.+)_resnet50(?:_(?:bbox_bg|histmatch))?$",
        experiment_name.lower(),
    )
    if not match:
        return None
    return match.group("scenario")


def build_intervention_rows(summary_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        experiment_name = summary["experiment_name"]
        scenario = infer_scenario(experiment_name)
        variant = infer_variant(experiment_name)
        if scenario is None or variant is None:
            continue

        run_dir = str(summary_path.parent)
        run_row: dict[str, Any] = {
            "experiment_name": experiment_name,
            "scenario": scenario,
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "run_dir": run_dir,
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": summary.get("selection_metric"),
            "best_score": summary.get("best_score"),
        }

        for split_name, split_metrics in summary.items():
            if not isinstance(split_metrics, dict):
                continue
            if "accuracy" not in split_metrics or "macro_f1" not in split_metrics:
                continue

            run_row[f"{split_name}_accuracy"] = split_metrics["accuracy"]
            run_row[f"{split_name}_macro_f1"] = split_metrics["macro_f1"]
            split_rows.append(
                {
                    "experiment_name": experiment_name,
                    "scenario": scenario,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "run_dir": run_dir,
                    "split": split_name,
                    "accuracy": split_metrics["accuracy"],
                    "macro_f1": split_metrics["macro_f1"],
                }
            )

        run_rows.append(run_row)

    run_columns = [
        "experiment_name",
        "scenario",
        "variant",
        "variant_label",
        "run_dir",
        "best_epoch",
        "selection_metric",
        "best_score",
    ]
    split_columns = [
        "experiment_name",
        "scenario",
        "variant",
        "variant_label",
        "run_dir",
        "split",
        "accuracy",
        "macro_f1",
    ]
    runs_df = pd.DataFrame(run_rows)
    splits_df = pd.DataFrame(split_rows)
    if runs_df.empty:
        runs_df = pd.DataFrame(columns=run_columns)
    else:
        runs_df["variant"] = pd.Categorical(runs_df["variant"], categories=VARIANT_ORDER, ordered=True)
        runs_df = runs_df.sort_values(["scenario", "variant", "experiment_name"]).reset_index(drop=True)
    if splits_df.empty:
        splits_df = pd.DataFrame(columns=split_columns)
    else:
        splits_df["variant"] = pd.Categorical(splits_df["variant"], categories=VARIANT_ORDER, ordered=True)
        splits_df = splits_df.sort_values(["scenario", "variant", "split"]).reset_index(drop=True)
    return runs_df, splits_df


def build_intervention_metrics(splits_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "variant",
        "variant_label",
        "in_domain_accuracy",
        "out_of_domain_accuracy",
        "gap_accuracy",
        "normalized_gap",
    ]
    if splits_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = splits_df.groupby(["scenario", "variant", "variant_label"], sort=True)

    for (scenario, variant, variant_label), frame in grouped:
        classified = frame.copy()
        classified["domain_role"] = classified["split"].map(
            lambda split_name: classify_split_domain(split_name, scenario)
        )

        in_domain = classified[classified["domain_role"] == "in"]
        out_of_domain = classified[classified["domain_role"] == "out"]
        if in_domain.empty or out_of_domain.empty:
            continue

        in_domain_accuracy = float(in_domain["accuracy"].mean())
        out_of_domain_accuracy = float(out_of_domain["accuracy"].mean())
        gap_accuracy = in_domain_accuracy - out_of_domain_accuracy
        normalized_gap = gap_accuracy / in_domain_accuracy if in_domain_accuracy else None

        rows.append(
            {
                "scenario": scenario,
                "variant": variant,
                "variant_label": variant_label,
                "in_domain_accuracy": in_domain_accuracy,
                "out_of_domain_accuracy": out_of_domain_accuracy,
                "gap_accuracy": gap_accuracy,
                "normalized_gap": normalized_gap,
            }
        )

    metrics_df = pd.DataFrame(rows, columns=columns)
    if metrics_df.empty:
        return metrics_df
    metrics_df["variant"] = pd.Categorical(metrics_df["variant"], categories=VARIANT_ORDER, ordered=True)
    return metrics_df.sort_values(["scenario", "variant"]).reset_index(drop=True)


def write_markdown(
    output_path: Path,
    runs_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    lines = [
        "# ResNet50 Data Intervention Comparison",
        "",
        "This table compares ResNet50 runs across dataset variants for the same camera-trap scenarios.",
        "",
    ]

    if runs_df.empty:
        lines.extend(["No intervention summary files were found.", ""])
    else:
        lines.extend(["## Run Summary", "", dataframe_to_markdown(runs_df), ""])

    if metrics_df.empty:
        lines.extend(["## Generalization Metrics", "", "No intervention metrics were produced.", ""])
    else:
        lines.extend(
            [
                "## Generalization Metrics",
                "",
                dataframe_to_markdown(metrics_df),
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_summary_paths = find_summary_files(results_root)
    summary_paths = select_preferred_summary_paths(discovered_summary_paths)
    runs_df, splits_df = build_intervention_rows(summary_paths)
    metrics_df = build_intervention_metrics(splits_df)

    runs_df.to_csv(output_dir / "intervention_runs.csv", index=False)
    splits_df.to_csv(output_dir / "intervention_split_metrics.csv", index=False)
    metrics_df.to_csv(output_dir / "intervention_metrics.csv", index=False)
    write_markdown(output_dir / "intervention_comparison.md", runs_df, metrics_df)

    payload = {
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "discovered_summary_file_count": len(discovered_summary_paths),
        "selected_summary_file_count": len(summary_paths),
        "runs_csv": str(output_dir / "intervention_runs.csv"),
        "split_metrics_csv": str(output_dir / "intervention_split_metrics.csv"),
        "metrics_csv": str(output_dir / "intervention_metrics.csv"),
        "markdown_report": str(output_dir / "intervention_comparison.md"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
