#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ResNet-18/ResNet-50 experiment summaries for capacity comparisons."
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_backbone(experiment_name: str) -> str:
    lowered = experiment_name.lower()
    if "resnet18" in lowered:
        return "resnet18"
    if "resnet50" in lowered:
        return "resnet50"
    return "unknown"


def infer_scenario(experiment_name: str) -> str:
    return re.sub(r"_resnet(18|50)$", "", experiment_name)


def find_summary_files(results_root: Path) -> list[Path]:
    return sorted(path for path in results_root.glob("*/summary.json") if path.is_file())


def select_preferred_summary_paths(summary_paths: list[Path]) -> list[Path]:
    selected: dict[str, tuple[float, int, str, Path]] = {}

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        experiment_name = summary["experiment_name"]
        best_score = float(summary.get("best_score", float("-inf")))
        best_epoch = int(summary.get("best_epoch", -1))
        run_dir = str(summary_path.parent)
        candidate = (best_score, best_epoch, run_dir, summary_path)

        current = selected.get(experiment_name)
        if current is None or candidate[:3] > current[:3]:
            selected[experiment_name] = candidate

    return sorted(candidate[3] for candidate in selected.values())


def build_run_rows(summary_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        experiment_name = summary["experiment_name"]
        scenario = infer_scenario(experiment_name)
        backbone = infer_backbone(experiment_name)
        run_dir = str(summary_path.parent)

        run_row: dict[str, Any] = {
            "experiment_name": experiment_name,
            "scenario": scenario,
            "backbone": backbone,
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
                    "backbone": backbone,
                    "run_dir": run_dir,
                    "split": split_name,
                    "accuracy": split_metrics["accuracy"],
                    "macro_f1": split_metrics["macro_f1"],
                }
            )

        run_rows.append(run_row)

    runs_df = pd.DataFrame(run_rows).sort_values(["scenario", "backbone", "experiment_name"]).reset_index(drop=True)
    splits_df = pd.DataFrame(split_rows).sort_values(["scenario", "split", "backbone"]).reset_index(drop=True)
    return runs_df, splits_df


def build_capacity_delta_table(splits_df: pd.DataFrame) -> pd.DataFrame:
    if splits_df.empty:
        return pd.DataFrame()

    pivot = splits_df.pivot_table(
        index=["scenario", "split"],
        columns="backbone",
        values=["accuracy", "macro_f1"],
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()

    pivot.columns = [f"{metric}_{backbone}" for metric, backbone in pivot.columns]
    pivot = pivot.reset_index()

    if {"accuracy_resnet18", "accuracy_resnet50"}.issubset(pivot.columns):
        pivot["accuracy_delta_resnet50_minus_resnet18"] = (
            pivot["accuracy_resnet50"] - pivot["accuracy_resnet18"]
        )
    if {"macro_f1_resnet18", "macro_f1_resnet50"}.issubset(pivot.columns):
        pivot["macro_f1_delta_resnet50_minus_resnet18"] = (
            pivot["macro_f1_resnet50"] - pivot["macro_f1_resnet18"]
        )

    return pivot.sort_values(["scenario", "split"]).reset_index(drop=True)


def infer_transfer_domains(scenario: str) -> tuple[str | None, str | None]:
    match = re.search(r"([a-z0-9]+)_to_([a-z0-9]+)", scenario.lower())
    if not match:
        return None, None
    return match.group(1), match.group(2)


def classify_split_domain(split_name: str, scenario: str) -> str | None:
    split_lower = split_name.lower()
    tokens = set(re.findall(r"[a-z0-9]+", split_lower))
    source_domain, target_domain = infer_transfer_domains(scenario)

    if target_domain and target_domain in tokens:
        return "out"
    if source_domain and source_domain in tokens:
        return "in"

    in_keywords = {"train", "val", "in", "source", "cis"}
    out_keywords = {"test", "ood", "target", "out", "trans"}

    has_in_keyword = bool(tokens & in_keywords)
    has_out_keyword = bool(tokens & out_keywords)

    if has_in_keyword and not has_out_keyword:
        return "in"
    if has_out_keyword and not has_in_keyword:
        return "out"
    return None


def build_generalization_drop_table(splits_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "backbone",
        "in_domain_accuracy",
        "out_of_domain_accuracy",
        "drop_accuracy",
        "relative_drop",
    ]
    if splits_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = splits_df.groupby(["scenario", "backbone"], sort=True)

    for (scenario, backbone), frame in grouped:
        classified = frame.copy()
        classified["domain_role"] = classified["split"].map(lambda split_name: classify_split_domain(split_name, scenario))

        in_domain = classified[classified["domain_role"] == "in"]
        out_of_domain = classified[classified["domain_role"] == "out"]

        if in_domain.empty or out_of_domain.empty:
            continue

        in_domain_accuracy = float(in_domain["accuracy"].mean())
        out_of_domain_accuracy = float(out_of_domain["accuracy"].mean())
        drop_accuracy = in_domain_accuracy - out_of_domain_accuracy
        relative_drop = drop_accuracy / in_domain_accuracy if in_domain_accuracy else None

        rows.append(
            {
                "scenario": scenario,
                "backbone": backbone,
                "in_domain_accuracy": in_domain_accuracy,
                "out_of_domain_accuracy": out_of_domain_accuracy,
                "drop_accuracy": drop_accuracy,
                "relative_drop": relative_drop,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(["scenario", "backbone"]).reset_index(drop=True)


def build_drop_comparison_table(drop_df: pd.DataFrame) -> pd.DataFrame:
    if drop_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "in_domain_accuracy_resnet18",
                "out_of_domain_accuracy_resnet18",
                "drop_accuracy_resnet18",
                "relative_drop_resnet18",
                "in_domain_accuracy_resnet50",
                "out_of_domain_accuracy_resnet50",
                "drop_accuracy_resnet50",
                "relative_drop_resnet50",
                "drop_delta",
                "relative_drop_delta",
            ]
        )

    pivot = drop_df.pivot_table(
        index="scenario",
        columns="backbone",
        values=["in_domain_accuracy", "out_of_domain_accuracy", "drop_accuracy", "relative_drop"],
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()

    pivot.columns = [f"{metric}_{backbone}" for metric, backbone in pivot.columns]
    pivot = pivot.reset_index()

    if {"drop_accuracy_resnet18", "drop_accuracy_resnet50"}.issubset(pivot.columns):
        pivot["drop_delta"] = pivot["drop_accuracy_resnet50"] - pivot["drop_accuracy_resnet18"]
    if {"relative_drop_resnet18", "relative_drop_resnet50"}.issubset(pivot.columns):
        pivot["relative_drop_delta"] = pivot["relative_drop_resnet50"] - pivot["relative_drop_resnet18"]

    return pivot.sort_values(["scenario"]).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    normalized = df.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(lambda value: "" if pd.isna(value) else str(value))

    headers = [str(column) for column in normalized.columns]
    rows = normalized.values.tolist()
    widths = [
        max(len(headers[index]), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(values: list[str]) -> str:
        cells = [str(value).ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [render_row(headers), separator]
    lines.extend(render_row([str(value) for value in row]) for row in rows)
    return "\n".join(lines)


def write_markdown(
    output_path: Path,
    runs_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    lines = [
        "# Capacity Comparison",
        "",
        "This table compares ResNet-18 and ResNet-50 runs for the same CCT20 experiment scenarios.",
        "",
    ]

    if runs_df.empty:
        lines.extend(["No summary files were found.", ""])
    else:
        lines.extend(["## Run Summary", "", dataframe_to_markdown(runs_df), ""])

    if delta_df.empty:
        lines.extend(["## Capacity Delta", "", "Not enough paired ResNet-18/ResNet-50 runs were found yet.", ""])
    else:
        lines.extend(
            [
                "## Capacity Delta",
                "",
                "Positive deltas mean ResNet-50 outperformed ResNet-18 on that split.",
                "",
                dataframe_to_markdown(delta_df),
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
    runs_df, splits_df = build_run_rows(summary_paths)
    delta_df = build_capacity_delta_table(splits_df)
    drop_df = build_generalization_drop_table(splits_df)
    drop_comparison_df = build_drop_comparison_table(drop_df)

    runs_df.to_csv(output_dir / "capacity_runs.csv", index=False)
    splits_df.to_csv(output_dir / "capacity_split_metrics.csv", index=False)
    delta_df.to_csv(output_dir / "capacity_deltas.csv", index=False)
    drop_comparison_df.to_csv(output_dir / "capacity_drop_comparison.csv", index=False)
    write_markdown(output_dir / "capacity_comparison.md", runs_df, delta_df)

    payload = {
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "summary_file_count": len(summary_paths),
        "discovered_summary_file_count": len(discovered_summary_paths),
        "runs_csv": str(output_dir / "capacity_runs.csv"),
        "split_metrics_csv": str(output_dir / "capacity_split_metrics.csv"),
        "delta_csv": str(output_dir / "capacity_deltas.csv"),
        "drop_comparison_csv": str(output_dir / "capacity_drop_comparison.csv"),
        "markdown_report": str(output_dir / "capacity_comparison.md"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
