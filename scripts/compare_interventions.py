#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from compare_capacity import (
    classify_split_domain,
    dataframe_to_markdown,
    find_summary_files,
    load_json,
)


VARIANT_LABELS = {
    "original": "Original",
    "photometric_randomization": "Photometric Randomization",
    "background_perturbation": "Background Perturbation",
    "combined": "Combined",
}

SCENARIO_ORDER = ["cross_location", "day_to_night", "night_to_day"]
BACKBONE_ORDER = ["resnet18", "resnet34", "resnet50", "resnet101"]
VARIANT_ORDER = ["original", "photometric_randomization", "background_perturbation", "combined"]
EXPECTED_COLUMNS = ["scenario", "backbone", "variant", "expected_config", "expected_train_intervention"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate train-time intervention experiment summaries."
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


def variant_suffix(variant: str) -> str:
    if variant == "original":
        return ""
    return f"_{variant}"


def expected_train_intervention(variant: str) -> str:
    return "none" if variant == "original" else variant


def build_expected_matrix() -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        for backbone in BACKBONE_ORDER:
            for variant in VARIANT_ORDER:
                rows.append(
                    {
                        "scenario": scenario,
                        "backbone": backbone,
                        "variant": variant,
                        "expected_config": f"configs/{scenario}_{backbone}.yaml",
                        "expected_train_intervention": expected_train_intervention(variant),
                    }
                )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def infer_variant(experiment_name: str) -> str | None:
    name = experiment_name.lower()
    if not re.match(r"(cross_location|day_to_night|night_to_day)_resnet(18|34|50|101)", name):
        return None
    for variant in VARIANT_ORDER:
        if variant == "original":
            continue
        if name.endswith(f"_{variant}"):
            return variant
    if re.search(r"_resnet(18|34|50|101)$", name):
        return "original"
    return None


def infer_scenario(experiment_name: str) -> str | None:
    match = re.match(
        r"(?P<scenario>.+)_resnet(18|34|50|101)(?:_(?:photometric_randomization|background_perturbation|combined))?$",
        experiment_name.lower(),
    )
    if not match:
        return None
    return match.group("scenario")


def infer_backbone(experiment_name: str) -> str | None:
    match = re.search(r"resnet(18|34|50|101)", experiment_name.lower())
    if not match:
        return None
    return f"resnet{match.group(1)}"


def infer_depth(backbone: str | None) -> int | None:
    if backbone is None:
        return None
    match = re.search(r"(\d+)$", backbone)
    return int(match.group(1)) if match else None


def fallback_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def build_run_notes(dataset_summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    notes: list[str] = []
    sanity_checks = dataset_summary.get("sanity_checks", {})
    if sanity_checks:
        failed_checks = [name for name, passed in sanity_checks.items() if not passed]
        if failed_checks:
            notes.append(f"failed sanity checks: {', '.join(failed_checks)}")
        else:
            notes.append("sanity checks passed")

    manifest_notes = manifest.get("notes")
    if manifest_notes:
        notes.append(str(manifest_notes))

    return "; ".join(notes) if notes else "no run manifest available"


def select_latest_summary_paths(summary_paths: list[Path]) -> list[Path]:
    selected: dict[str, tuple[str, float, Path]] = {}
    for summary_path in summary_paths:
        summary = load_json(summary_path)
        experiment_name = summary["experiment_name"]
        run_key = summary_path.parent.name
        modified_time = summary_path.stat().st_mtime
        candidate = (run_key, modified_time, summary_path)
        current = selected.get(experiment_name)
        if current is None or candidate[:2] > current[:2]:
            selected[experiment_name] = candidate
    return sorted(candidate[2] for candidate in selected.values())


def build_intervention_rows(summary_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        experiment_name = summary["experiment_name"]
        scenario = infer_scenario(experiment_name)
        variant = infer_variant(experiment_name)
        backbone = infer_backbone(experiment_name)
        if scenario is None or variant is None or backbone is None:
            continue

        run_dir_path = summary_path.parent
        run_dir = str(run_dir_path)
        manifest = load_optional_json(run_dir_path / "run_manifest.json")
        resolved_config = load_optional_json(run_dir_path / "resolved_config.json")
        dataset_summary = load_optional_json(run_dir_path / "dataset_summary.json")
        seed = summary.get("seed", manifest.get("seed", resolved_config.get("seed")))
        run_id = summary.get("run_id", manifest.get("run_id", run_dir_path.name))
        timestamp = summary.get(
            "timestamp",
            manifest.get("timestamp", dataset_summary.get("timestamp", fallback_timestamp(summary_path))),
        )
        notes = build_run_notes(dataset_summary, manifest)
        sanity_checks = dataset_summary.get("sanity_checks", {})
        split_interventions = dataset_summary.get("split_interventions", {})
        run_row: dict[str, Any] = {
            "experiment_name": experiment_name,
            "scenario": scenario,
            "backbone": backbone,
            "depth": infer_depth(backbone),
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "seed": seed,
            "run_id": run_id,
            "timestamp": timestamp,
            "run_dir": run_dir,
            "config_path": resolved_config.get("_config_path", manifest.get("config_path")),
            "command": manifest.get("command"),
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": summary.get("selection_metric"),
            "best_score": summary.get("best_score"),
            "split_interventions": json.dumps(split_interventions, sort_keys=True),
            "train_intervention_only": sanity_checks.get("train_intervention_only"),
            "validation_and_test_images_unmodified_by_intervention": sanity_checks.get(
                "validation_and_test_images_unmodified_by_intervention"
            ),
            "test_domain_statistics_used": sanity_checks.get("test_domain_statistics_used"),
            "notes": notes,
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
                    "depth": infer_depth(backbone),
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "seed": seed,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "run_dir": run_dir,
                    "split": split_name,
                    "accuracy": split_metrics["accuracy"],
                    "macro_f1": split_metrics["macro_f1"],
                    "notes": notes,
                }
            )

        run_rows.append(run_row)

    run_columns = [
        "experiment_name",
        "scenario",
        "backbone",
        "depth",
        "variant",
        "variant_label",
        "seed",
        "run_id",
        "timestamp",
        "run_dir",
        "config_path",
        "command",
        "best_epoch",
        "selection_metric",
        "best_score",
        "split_interventions",
        "train_intervention_only",
        "validation_and_test_images_unmodified_by_intervention",
        "test_domain_statistics_used",
        "notes",
    ]
    split_columns = [
        "experiment_name",
        "scenario",
        "backbone",
        "depth",
        "variant",
        "variant_label",
        "seed",
        "run_id",
        "timestamp",
        "run_dir",
        "split",
        "accuracy",
        "macro_f1",
        "notes",
    ]
    runs_df = pd.DataFrame(run_rows)
    splits_df = pd.DataFrame(split_rows)
    if runs_df.empty:
        runs_df = pd.DataFrame(columns=run_columns)
    else:
        runs_df["variant"] = pd.Categorical(runs_df["variant"], categories=VARIANT_ORDER, ordered=True)
        runs_df["scenario"] = pd.Categorical(runs_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
        runs_df["backbone"] = pd.Categorical(runs_df["backbone"], categories=BACKBONE_ORDER, ordered=True)
        runs_df = runs_df.sort_values(["scenario", "backbone", "variant", "experiment_name"]).reset_index(drop=True)
    if splits_df.empty:
        splits_df = pd.DataFrame(columns=split_columns)
    else:
        splits_df["variant"] = pd.Categorical(splits_df["variant"], categories=VARIANT_ORDER, ordered=True)
        splits_df["scenario"] = pd.Categorical(splits_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
        splits_df["backbone"] = pd.Categorical(splits_df["backbone"], categories=BACKBONE_ORDER, ordered=True)
        splits_df = splits_df.sort_values(["scenario", "backbone", "variant", "split"]).reset_index(drop=True)
    return runs_df, splits_df


def build_intervention_metrics(splits_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "backbone",
        "scenario",
        "variant",
        "variant_label",
        "in_domain_acc",
        "ood_acc",
        "gap",
        "normalized_gap",
        "seed",
        "run_id",
        "timestamp",
        "notes",
    ]
    if splits_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = splits_df.groupby(
        ["scenario", "backbone", "depth", "variant", "variant_label", "seed", "run_id", "timestamp", "notes"],
        sort=True,
        observed=True,
        dropna=False,
    )

    for (scenario, backbone, _depth, variant, variant_label, seed, run_id, timestamp, notes), frame in grouped:
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
                "backbone": backbone,
                "scenario": scenario,
                "variant": variant,
                "variant_label": variant_label,
                "in_domain_acc": in_domain_accuracy,
                "ood_acc": out_of_domain_accuracy,
                "gap": gap_accuracy,
                "normalized_gap": normalized_gap,
                "seed": seed,
                "run_id": run_id,
                "timestamp": timestamp,
                "notes": notes,
            }
        )

    metrics_df = pd.DataFrame(rows, columns=columns)
    if metrics_df.empty:
        return metrics_df
    metrics_df["variant"] = pd.Categorical(metrics_df["variant"], categories=VARIANT_ORDER, ordered=True)
    metrics_df["scenario"] = pd.Categorical(metrics_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    metrics_df["backbone"] = pd.Categorical(metrics_df["backbone"], categories=BACKBONE_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values(["scenario", "backbone", "variant"]).reset_index(drop=True)
    return metrics_df


def build_matrix_status(expected_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "scenario",
        "backbone",
        "variant",
        "in_domain_acc",
        "ood_acc",
        "gap",
        "normalized_gap",
        "seed",
        "run_id",
        "timestamp",
        "notes",
    ]
    if metrics_df.empty:
        status_df = expected_df.copy()
        for column in metric_columns:
            if column not in status_df.columns:
                status_df[column] = pd.NA
        status_df["status"] = "missing"
        return status_df[
            EXPECTED_COLUMNS
            + ["status", "in_domain_acc", "ood_acc", "gap", "normalized_gap", "seed", "run_id", "timestamp", "notes"]
        ]

    completed = metrics_df[metric_columns].copy()
    status_df = expected_df.merge(completed, on=["scenario", "backbone", "variant"], how="left")
    status_df["status"] = status_df["run_id"].map(lambda value: "completed" if pd.notna(value) else "missing")
    return status_df[
        EXPECTED_COLUMNS
        + ["status", "in_domain_acc", "ood_acc", "gap", "normalized_gap", "seed", "run_id", "timestamp", "notes"]
    ]


def build_effect_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "backbone",
        "variant",
        "variant_label",
        "original_in_domain_acc",
        "in_domain_acc",
        "delta_in_domain_acc",
        "original_ood_acc",
        "ood_acc",
        "delta_ood_acc",
        "original_normalized_gap",
        "normalized_gap",
        "delta_normalized_gap",
        "seed",
        "run_id",
        "timestamp",
        "notes",
    ]
    if metrics_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (scenario, backbone), frame in metrics_df.groupby(["scenario", "backbone"], observed=True, sort=False):
        original = frame[frame["variant"] == "original"]
        if original.empty:
            continue
        baseline = original.iloc[0]
        for _, row in frame.iterrows():
            rows.append(
                {
                    "scenario": scenario,
                    "backbone": backbone,
                    "variant": row["variant"],
                    "variant_label": row["variant_label"],
                    "original_in_domain_acc": baseline["in_domain_acc"],
                    "in_domain_acc": row["in_domain_acc"],
                    "delta_in_domain_acc": row["in_domain_acc"] - baseline["in_domain_acc"],
                    "original_ood_acc": baseline["ood_acc"],
                    "ood_acc": row["ood_acc"],
                    "delta_ood_acc": row["ood_acc"] - baseline["ood_acc"],
                    "original_normalized_gap": baseline["normalized_gap"],
                    "normalized_gap": row["normalized_gap"],
                    "delta_normalized_gap": row["normalized_gap"] - baseline["normalized_gap"],
                    "seed": row["seed"],
                    "run_id": row["run_id"],
                    "timestamp": row["timestamp"],
                    "notes": row["notes"],
                }
            )

    effect_df = pd.DataFrame(rows, columns=columns)
    if effect_df.empty:
        return effect_df
    effect_df["scenario"] = pd.Categorical(effect_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    effect_df["backbone"] = pd.Categorical(effect_df["backbone"], categories=BACKBONE_ORDER, ordered=True)
    effect_df["variant"] = pd.Categorical(effect_df["variant"], categories=VARIANT_ORDER, ordered=True)
    return effect_df.sort_values(["scenario", "backbone", "variant"]).reset_index(drop=True)


def format_float(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def answer_comparison_questions(metrics_df: pd.DataFrame, effect_df: pd.DataFrame) -> list[str]:
    if metrics_df.empty:
        return [
            "尚未发现已完成的训练结果，因此目前不能回答效果对比问题。",
            "所有期望组合都已写入 missing_runs.csv，可据此在 CARC 上提交或补收集结果。",
        ]

    answers: list[str] = []
    non_original = effect_df[effect_df["variant"] != "original"].copy()

    cross_location = non_original[non_original["scenario"] == "cross_location"]
    if cross_location.empty:
        answers.append("cross_location：没有可配对的干预结果。")
    else:
        grouped = cross_location.groupby("variant", observed=True)["delta_ood_acc"].mean().sort_values(ascending=False)
        best_variant = grouped.index[0]
        answers.append(
            f"cross_location：{VARIANT_LABELS[str(best_variant)]} 对平均 OOD accuracy 帮助最大，"
            f"相对 original 变化为 {format_float(grouped.iloc[0])}。"
        )

    day_to_night = non_original[non_original["scenario"] == "day_to_night"]
    if day_to_night.empty:
        answers.append("day_to_night：没有可配对的干预结果。")
    else:
        grouped = day_to_night.groupby("variant", observed=True)["delta_ood_acc"].mean().sort_values(ascending=False)
        best_variant = grouped.index[0]
        worst_variant = grouped.index[-1]
        answers.append(
            f"day_to_night：{VARIANT_LABELS[str(best_variant)]} 最有帮助 "
            f"({format_float(grouped.iloc[0])} vs original)，"
            f"{VARIANT_LABELS[str(worst_variant)]} 最不利 "
            f"({format_float(grouped.iloc[-1])} vs original)。"
        )

    stable = (
        metrics_df.groupby("backbone", observed=True)["normalized_gap"]
        .agg(["std", "mean", "count"])
        .reset_index()
    )
    stable = stable[stable["count"] > 1].sort_values(["std", "mean"], ascending=[True, True])
    if stable.empty:
        answers.append("backbone 稳定性：已完成结果不足，暂时无法估计跨场景稳定性。")
    else:
        row = stable.iloc[0]
        answers.append(
            f"backbone 稳定性：按 normalized gap 的标准差衡量，{row['backbone']} 最稳定 "
            f"(std {format_float(row['std'])}, mean {format_float(row['mean'])})。"
        )

    helpful = non_original[
        (non_original["delta_ood_acc"] > 0) & (non_original["delta_in_domain_acc"] >= -0.02)
    ].copy()
    if helpful.empty:
        answers.append("trade-off：没有干预在提升 OOD accuracy 的同时把 in-domain 损失控制在 0.020 以内。")
    else:
        helpful = helpful.sort_values("delta_ood_acc", ascending=False)
        top = helpful.iloc[0]
        answers.append(
            "trade-off：至少有一个干预在较小 in-domain 损失下提升了 OOD accuracy；"
            f"最强的是 {top['scenario']} / {top['backbone']} 上的 {VARIANT_LABELS[str(top['variant'])]} "
            f"(delta OOD {format_float(top['delta_ood_acc'])}, delta in-domain {format_float(top['delta_in_domain_acc'])})。"
        )

    if non_original.empty:
        answers.append("normalized gap：没有可配对的干预结果。")
    else:
        consistency_rows = []
        for variant, frame in non_original.groupby("variant", observed=True):
            improved = int((frame["delta_normalized_gap"] < 0).sum())
            total = int(frame["delta_normalized_gap"].notna().sum())
            consistency_rows.append(f"{VARIANT_LABELS[str(variant)]}: {improved}/{total}")
        answers.append(
            "normalized gap 是否稳定下降可看 improved/available 计数："
            + "; ".join(consistency_rows)
            + "."
        )

    return answers


def write_markdown(
    output_path: Path,
    runs_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    matrix_status_df: pd.DataFrame,
    effect_df: pd.DataFrame,
) -> None:
    completed_count = int((matrix_status_df["status"] == "completed").sum()) if not matrix_status_df.empty else 0
    missing_count = int((matrix_status_df["status"] == "missing").sum()) if not matrix_status_df.empty else 0
    lines = [
        "# Train-Time Diversification Comparison",
        "",
        "This report compares train-only distribution-diversification interventions across backbones and camera-trap scenarios.",
        "",
        "Design rationale: stochastic augmentation increases training-domain appearance diversity instead of forcing all inputs toward one aligned appearance.",
        "",
        "## Matrix Status",
        "",
        f"- Expected combinations: {len(matrix_status_df)}",
        f"- Completed combinations: {completed_count}",
        f"- Missing combinations: {missing_count}",
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

    lines.extend(["## 中文结果摘要", ""])
    for answer in answer_comparison_questions(metrics_df, effect_df):
        lines.append(f"- {answer}")
    lines.append("")

    missing_df = matrix_status_df[matrix_status_df["status"] == "missing"]
    if not missing_df.empty:
        lines.extend(
            [
                "## Missing Runs",
                "",
                "Missing combinations are written to `missing_runs.csv`; submit or collect those runs before interpreting the matrix.",
                "",
                dataframe_to_markdown(missing_df[EXPECTED_COLUMNS].head(48)),
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_run_manifest(output_path: Path, output_dir: Path, matrix_status_df: pd.DataFrame) -> None:
    completed_count = int((matrix_status_df["status"] == "completed").sum()) if not matrix_status_df.empty else 0
    lines = [
        "# Train-Time Diversification Run Manifest",
        "",
        "## Experiment Matrix",
        "",
        "- Dataset variants: original, photometric_randomization, background_perturbation, combined",
        "- Scenarios: cross_location, day_to_night, night_to_day",
        "- Backbones: resnet18, resnet34, resnet50, resnet101",
        "- Seed: read from each run config, default baseline configs use 42",
        "- Optimizer/epochs/batch size/augmentation: held constant by the shared YAML training block unless explicitly changed in a config",
        "- Train-time intervention scope: train split only; validation and test splits must report intervention `none` in `dataset_summary.json`",
        "- No validation or test-domain statistics are used by the active variants",
        "- Stochastic augmentations are logged in each run's `dataset_summary.json` and `run_manifest.json`",
        "",
        "## Output Files",
        "",
        f"- Output directory: `{output_dir}`",
        "- `intervention_runs.csv`: one row per selected run directory",
        "- `intervention_split_metrics.csv`: one row per evaluated split",
        "- `intervention_metrics.csv`: tidy in-domain/OOD/gap table",
        "- `intervention_effect_metrics.csv`: deltas against the original variant",
        "- `experiment_matrix.csv`: expected 48-combination matrix with completion status",
        "- `missing_runs.csv`: combinations not found in the results root",
        "- `intervention_comparison.md`: concise answers for the report questions",
        "- `intervention_summary_zh.md`: Chinese summary of the comparison questions",
        "- `sanity_checks.json`: aggregation-level sanity checks",
        "- Plot PNGs: OOD accuracy, normalized gap, deltas, scatter, backbone lines, and grid views",
        "",
        "## Current Status",
        "",
        f"- Completed combinations in this aggregation: {completed_count}/48",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_chinese_summary(output_path: Path, metrics_df: pd.DataFrame, effect_df: pd.DataFrame) -> None:
    lines = [
        "# 训练期分布多样化结果摘要",
        "",
        "这版实验使用随机增强来扩大训练域外观多样性，而不是把图像强行对齐到单一外观。所有干预都只作用于训练 split，validation/test 保持原始图像与原始评估流程。",
        "",
    ]
    for answer in answer_comparison_questions(metrics_df, effect_df):
        lines.append(f"- {answer}")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_sanity_checks(matrix_status_df: pd.DataFrame, runs_df: pd.DataFrame) -> dict[str, Any]:
    completed_count = int((matrix_status_df["status"] == "completed").sum()) if not matrix_status_df.empty else 0
    missing_count = int((matrix_status_df["status"] == "missing").sum()) if not matrix_status_df.empty else 0
    if runs_df.empty:
        test_data_unchanged: bool | str = "no_completed_runs_to_verify"
        eval_transforms_disabled: bool | str = "no_completed_runs_to_verify"
        no_test_domain_stats: bool | str = "no_completed_runs_to_verify"
    else:
        test_data_unchanged = bool(
            runs_df["validation_and_test_images_unmodified_by_intervention"].fillna(False).astype(bool).all()
        )
        eval_transforms_disabled = bool(runs_df["train_intervention_only"].fillna(False).astype(bool).all())
        no_test_domain_stats = bool((~runs_df["test_domain_statistics_used"].fillna(True).astype(bool)).all())

    return {
        "expected_combination_count": int(len(matrix_status_df)),
        "completed_combination_count": completed_count,
        "missing_combination_count": missing_count,
        "all_expected_combinations_present_in_summary_csv": missing_count == 0,
        "test_data_unchanged": test_data_unchanged,
        "train_only_transforms_not_active_in_eval_mode": eval_transforms_disabled,
        "test_domain_statistics_not_used": no_test_domain_stats,
        "run_rows_seen": int(len(runs_df)),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_summary_paths = find_summary_files(results_root)
    summary_paths = select_latest_summary_paths(discovered_summary_paths)
    runs_df, splits_df = build_intervention_rows(summary_paths)
    metrics_df = build_intervention_metrics(splits_df)
    expected_df = build_expected_matrix()
    matrix_status_df = build_matrix_status(expected_df, metrics_df)
    missing_runs_df = matrix_status_df[matrix_status_df["status"] == "missing"].copy()
    effect_df = build_effect_metrics(metrics_df)

    runs_df.to_csv(output_dir / "intervention_runs.csv", index=False)
    splits_df.to_csv(output_dir / "intervention_split_metrics.csv", index=False)
    metrics_df.to_csv(output_dir / "intervention_metrics.csv", index=False)
    effect_df.to_csv(output_dir / "intervention_effect_metrics.csv", index=False)
    matrix_status_df.to_csv(output_dir / "experiment_matrix.csv", index=False)
    missing_runs_df.to_csv(output_dir / "missing_runs.csv", index=False)
    write_markdown(output_dir / "intervention_comparison.md", runs_df, metrics_df, matrix_status_df, effect_df)
    write_chinese_summary(output_dir / "intervention_summary_zh.md", metrics_df, effect_df)
    write_run_manifest(output_dir / "RUN_MANIFEST.md", output_dir, matrix_status_df)
    sanity_checks = build_sanity_checks(matrix_status_df, runs_df)
    (output_dir / "sanity_checks.json").write_text(json.dumps(sanity_checks, indent=2), encoding="utf-8")

    payload = {
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "discovered_summary_file_count": len(discovered_summary_paths),
        "selected_summary_file_count": len(summary_paths),
        "expected_combination_count": len(expected_df),
        "completed_combination_count": int((matrix_status_df["status"] == "completed").sum()),
        "missing_combination_count": int((matrix_status_df["status"] == "missing").sum()),
        "runs_csv": str(output_dir / "intervention_runs.csv"),
        "split_metrics_csv": str(output_dir / "intervention_split_metrics.csv"),
        "metrics_csv": str(output_dir / "intervention_metrics.csv"),
        "effect_metrics_csv": str(output_dir / "intervention_effect_metrics.csv"),
        "experiment_matrix_csv": str(output_dir / "experiment_matrix.csv"),
        "missing_runs_csv": str(output_dir / "missing_runs.csv"),
        "markdown_report": str(output_dir / "intervention_comparison.md"),
        "chinese_summary": str(output_dir / "intervention_summary_zh.md"),
        "run_manifest": str(output_dir / "RUN_MANIFEST.md"),
        "sanity_checks_json": str(output_dir / "sanity_checks.json"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
