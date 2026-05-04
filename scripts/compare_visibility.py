#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from compare_capacity import classify_split_domain, dataframe_to_markdown


SCENARIO_ORDER = ["cross_location", "day_to_night", "night_to_day"]
BACKBONE_ORDER = ["resnet18", "resnet34", "resnet50", "resnet101"]
SCOPE_ORDER = ["test_only", "train_test_consistent", "night_only"]
MODE_ORDER = ["original", "gamma", "clahe", "gamma_clahe"]
MODE_LABELS = {
    "original": "Original",
    "gamma": "Gamma",
    "clahe": "CLAHE",
    "gamma_clahe": "Gamma + CLAHE",
}
SCOPE_LABELS = {
    "test_only": "Test-time only",
    "train_test_consistent": "Train + test consistent",
    "night_only": "Night-only consistent",
}
VISIBILITY_PATTERN = re.compile(
    r"^(?P<scenario>cross_location|day_to_night|night_to_day)_"
    r"(?P<backbone>resnet18|resnet34|resnet50|resnet101)"
    r"_visibility_(?P<scope>test_only|train_test_consistent|night_only)_"
    r"(?P<mode>original|gamma|clahe|gamma_clahe)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate visibility hypothesis experiment summaries.")
    parser.add_argument(
        "--results-root",
        required=True,
        help="Directory containing run folders with summary.json outputs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where visibility comparison artifacts will be written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def fallback_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def find_summary_files(results_root: Path) -> list[Path]:
    return sorted(path for path in results_root.rglob("summary.json") if path.is_file())


def infer_visibility_fields(summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str] | None:
    experiment_name = str(summary.get("experiment_name", ""))
    match = VISIBILITY_PATTERN.match(experiment_name)
    if match:
        return match.groupdict()

    visibility = summary.get("visibility_preprocessing") or manifest.get("visibility_preprocessing") or {}
    scope = visibility.get("scope")
    mode = visibility.get("mode")
    scenario = manifest.get("scenario")
    backbone = manifest.get("backbone")
    if scenario in SCENARIO_ORDER and backbone in BACKBONE_ORDER and scope in SCOPE_ORDER and mode in MODE_ORDER:
        return {
            "scenario": scenario,
            "backbone": backbone,
            "scope": scope,
            "mode": mode,
        }
    return None


def select_latest_summary_paths(summary_paths: list[Path]) -> list[Path]:
    selected: dict[tuple[str, str, str, str], tuple[float, str, Path]] = {}
    for summary_path in summary_paths:
        try:
            summary = load_json(summary_path)
        except (json.JSONDecodeError, OSError):
            continue
        manifest = load_optional_json(summary_path.parent / "run_manifest.json")
        fields = infer_visibility_fields(summary, manifest)
        if fields is None:
            continue
        key = (fields["scenario"], fields["backbone"], fields["scope"], fields["mode"])
        candidate = (summary_path.stat().st_mtime, summary_path.parent.name, summary_path)
        current = selected.get(key)
        if current is None or candidate[:2] > current[:2]:
            selected[key] = candidate
    return sorted(candidate[2] for candidate in selected.values())


def build_expected_matrix() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for scenario in SCENARIO_ORDER:
        for backbone in BACKBONE_ORDER:
            for scope in SCOPE_ORDER:
                for mode in MODE_ORDER:
                    rows.append(
                        {
                            "scenario": scenario,
                            "backbone": backbone,
                            "scope": scope,
                            "mode": mode,
                            "expected_config": f"configs/{scenario}_{backbone}.yaml",
                        }
                    )
    return pd.DataFrame(rows)


def infer_depth(backbone: str) -> int | None:
    match = re.search(r"(\d+)$", backbone)
    return int(match.group(1)) if match else None


def infer_split_light(split_name: str, scenario: str) -> str:
    split_lower = split_name.lower()
    if "night" in split_lower:
        return "night"
    if "day" in split_lower:
        return "day"
    if scenario == "day_to_night" and split_lower == "val":
        return "day"
    if scenario == "night_to_day" and split_lower == "val":
        return "night"
    return "mixed_or_unknown"


def build_rows(summary_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        run_dir_path = summary_path.parent
        manifest = load_optional_json(run_dir_path / "run_manifest.json")
        dataset_summary = load_optional_json(run_dir_path / "dataset_summary.json")
        resolved_config = load_optional_json(run_dir_path / "resolved_config.json")
        fields = infer_visibility_fields(summary, manifest)
        if fields is None:
            continue

        scenario = fields["scenario"]
        backbone = fields["backbone"]
        scope = fields["scope"]
        mode = fields["mode"]
        visibility = summary.get("visibility_preprocessing") or manifest.get("visibility_preprocessing") or {}
        sanity_checks = dataset_summary.get("sanity_checks", {})
        run_id = summary.get("run_id", manifest.get("run_id", run_dir_path.name))
        timestamp = summary.get(
            "timestamp",
            manifest.get("timestamp", dataset_summary.get("timestamp", fallback_timestamp(summary_path))),
        )

        run_row: dict[str, Any] = {
            "experiment_name": summary.get("experiment_name"),
            "scenario": scenario,
            "backbone": backbone,
            "depth": infer_depth(backbone),
            "scope": scope,
            "scope_label": SCOPE_LABELS[scope],
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "seed": summary.get("seed", manifest.get("seed", resolved_config.get("seed"))),
            "run_id": run_id,
            "timestamp": timestamp,
            "run_dir": str(run_dir_path),
            "config_path": resolved_config.get("_config_path", manifest.get("config_path")),
            "command": manifest.get("command"),
            "eval_only": bool(summary.get("eval_only", manifest.get("eval_only", False))),
            "checkpoint": summary.get("checkpoint", manifest.get("source_checkpoint")),
            "source_experiment_name": summary.get("source_experiment_name"),
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": summary.get("selection_metric"),
            "best_score": summary.get("best_score"),
            "visibility_gamma": visibility.get("gamma"),
            "visibility_clahe_clip_limit": visibility.get("clahe_clip_limit"),
            "visibility_clahe_tile_grid_size": json.dumps(visibility.get("clahe_tile_grid_size")),
            "night_only_flag_source": visibility.get("night_only_flag_source"),
            "visibility_application_counts": json.dumps(visibility.get("application_counts", {}), sort_keys=True),
            "sanity_checks": json.dumps(sanity_checks, sort_keys=True),
        }

        for split_name, split_metrics in summary.items():
            if not isinstance(split_metrics, dict):
                continue
            if "accuracy" not in split_metrics or "macro_f1" not in split_metrics:
                continue
            domain_role = classify_split_domain(split_name, scenario)
            split_light = infer_split_light(split_name, scenario)
            run_row[f"{split_name}_accuracy"] = split_metrics["accuracy"]
            run_row[f"{split_name}_macro_f1"] = split_metrics["macro_f1"]
            split_rows.append(
                {
                    "experiment_name": summary.get("experiment_name"),
                    "scenario": scenario,
                    "backbone": backbone,
                    "depth": infer_depth(backbone),
                    "scope": scope,
                    "scope_label": SCOPE_LABELS[scope],
                    "mode": mode,
                    "mode_label": MODE_LABELS[mode],
                    "seed": run_row["seed"],
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "run_dir": str(run_dir_path),
                    "split": split_name,
                    "domain_role": domain_role,
                    "split_light": split_light,
                    "accuracy": split_metrics["accuracy"],
                    "macro_f1": split_metrics["macro_f1"],
                }
            )

        run_rows.append(run_row)

    runs_df = pd.DataFrame(run_rows)
    splits_df = pd.DataFrame(split_rows)
    sort_columns = ["scenario", "backbone", "scope", "mode"]
    if not runs_df.empty:
        runs_df = sort_visibility_frame(runs_df, sort_columns)
    if not splits_df.empty:
        splits_df = sort_visibility_frame(splits_df, sort_columns + ["split"])
    return runs_df, splits_df


def sort_visibility_frame(frame: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    sorted_frame = frame.copy()
    if "scenario" in sorted_frame:
        sorted_frame["scenario"] = pd.Categorical(sorted_frame["scenario"], categories=SCENARIO_ORDER, ordered=True)
    if "backbone" in sorted_frame:
        sorted_frame["backbone"] = pd.Categorical(sorted_frame["backbone"], categories=BACKBONE_ORDER, ordered=True)
    if "scope" in sorted_frame:
        sorted_frame["scope"] = pd.Categorical(sorted_frame["scope"], categories=SCOPE_ORDER, ordered=True)
    if "mode" in sorted_frame:
        sorted_frame["mode"] = pd.Categorical(sorted_frame["mode"], categories=MODE_ORDER, ordered=True)
    return sorted_frame.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def build_visibility_metrics(splits_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "backbone",
        "depth",
        "scope",
        "scope_label",
        "mode",
        "mode_label",
        "in_domain_accuracy",
        "ood_accuracy",
        "generalization_gap",
        "normalized_gap",
        "seed",
        "run_id",
        "timestamp",
        "run_dir",
        "eval_only",
    ]
    if splits_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = splits_df.groupby(
        ["scenario", "backbone", "depth", "scope", "scope_label", "mode", "mode_label", "seed", "run_id", "timestamp", "run_dir"],
        observed=True,
        sort=False,
        dropna=False,
    )
    for key, frame in grouped:
        scenario, backbone, depth, scope, scope_label, mode, mode_label, seed, run_id, timestamp, run_dir = key
        in_domain = frame[frame["domain_role"] == "in"]
        out_domain = frame[frame["domain_role"] == "out"]
        if in_domain.empty or out_domain.empty:
            continue
        in_domain_accuracy = float(in_domain["accuracy"].mean())
        ood_accuracy = float(out_domain["accuracy"].mean())
        gap = in_domain_accuracy - ood_accuracy
        normalized_gap = gap / in_domain_accuracy if in_domain_accuracy else None
        rows.append(
            {
                "scenario": scenario,
                "backbone": backbone,
                "depth": depth,
                "scope": scope,
                "scope_label": scope_label,
                "mode": mode,
                "mode_label": mode_label,
                "in_domain_accuracy": in_domain_accuracy,
                "ood_accuracy": ood_accuracy,
                "generalization_gap": gap,
                "normalized_gap": normalized_gap,
                "seed": seed,
                "run_id": run_id,
                "timestamp": timestamp,
                "run_dir": run_dir,
                "eval_only": bool(scope == "test_only"),
            }
        )
    metrics_df = pd.DataFrame(rows, columns=columns)
    if metrics_df.empty:
        return metrics_df
    return sort_visibility_frame(metrics_df, ["scenario", "backbone", "scope", "mode"])


def build_effect_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "backbone",
        "scope",
        "mode",
        "mode_label",
        "original_in_domain_accuracy",
        "in_domain_accuracy",
        "delta_in_domain_accuracy",
        "original_ood_accuracy",
        "ood_accuracy",
        "delta_ood_accuracy",
        "original_normalized_gap",
        "normalized_gap",
        "delta_normalized_gap",
        "seed",
        "run_id",
        "timestamp",
        "run_dir",
    ]
    if metrics_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (scenario, backbone, scope), frame in metrics_df.groupby(["scenario", "backbone", "scope"], observed=True, sort=False):
        original = frame[frame["mode"] == "original"]
        if original.empty:
            continue
        baseline = original.iloc[0]
        for _, row in frame.iterrows():
            rows.append(
                {
                    "scenario": scenario,
                    "backbone": backbone,
                    "scope": scope,
                    "mode": row["mode"],
                    "mode_label": row["mode_label"],
                    "original_in_domain_accuracy": baseline["in_domain_accuracy"],
                    "in_domain_accuracy": row["in_domain_accuracy"],
                    "delta_in_domain_accuracy": row["in_domain_accuracy"] - baseline["in_domain_accuracy"],
                    "original_ood_accuracy": baseline["ood_accuracy"],
                    "ood_accuracy": row["ood_accuracy"],
                    "delta_ood_accuracy": row["ood_accuracy"] - baseline["ood_accuracy"],
                    "original_normalized_gap": baseline["normalized_gap"],
                    "normalized_gap": row["normalized_gap"],
                    "delta_normalized_gap": row["normalized_gap"] - baseline["normalized_gap"],
                    "seed": row["seed"],
                    "run_id": row["run_id"],
                    "timestamp": row["timestamp"],
                    "run_dir": row["run_dir"],
                }
            )
    effect_df = pd.DataFrame(rows, columns=columns)
    if effect_df.empty:
        return effect_df
    return sort_visibility_frame(effect_df, ["scenario", "backbone", "scope", "mode"])


def build_split_effect_metrics(splits_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "backbone",
        "scope",
        "mode",
        "split",
        "domain_role",
        "split_light",
        "original_accuracy",
        "accuracy",
        "delta_accuracy",
        "original_macro_f1",
        "macro_f1",
        "delta_macro_f1",
        "run_id",
    ]
    if splits_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (scenario, backbone, scope, split), frame in splits_df.groupby(
        ["scenario", "backbone", "scope", "split"], observed=True, sort=False
    ):
        original = frame[frame["mode"] == "original"]
        if original.empty:
            continue
        baseline = original.iloc[0]
        for _, row in frame.iterrows():
            rows.append(
                {
                    "scenario": scenario,
                    "backbone": backbone,
                    "scope": scope,
                    "mode": row["mode"],
                    "split": split,
                    "domain_role": row["domain_role"],
                    "split_light": row["split_light"],
                    "original_accuracy": baseline["accuracy"],
                    "accuracy": row["accuracy"],
                    "delta_accuracy": row["accuracy"] - baseline["accuracy"],
                    "original_macro_f1": baseline["macro_f1"],
                    "macro_f1": row["macro_f1"],
                    "delta_macro_f1": row["macro_f1"] - baseline["macro_f1"],
                    "run_id": row["run_id"],
                }
            )
    split_effect_df = pd.DataFrame(rows, columns=columns)
    if split_effect_df.empty:
        return split_effect_df
    return sort_visibility_frame(split_effect_df, ["scenario", "backbone", "scope", "mode", "split"])


def build_matrix_status(expected_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        status_df = expected_df.copy()
        status_df["status"] = "missing"
        for column in ["in_domain_accuracy", "ood_accuracy", "generalization_gap", "normalized_gap", "run_id", "timestamp"]:
            status_df[column] = pd.NA
        return status_df

    completed = metrics_df[
        [
            "scenario",
            "backbone",
            "scope",
            "mode",
            "in_domain_accuracy",
            "ood_accuracy",
            "generalization_gap",
            "normalized_gap",
            "run_id",
            "timestamp",
        ]
    ].copy()
    status_df = expected_df.merge(completed, on=["scenario", "backbone", "scope", "mode"], how="left")
    status_df["status"] = status_df["run_id"].map(lambda value: "completed" if pd.notna(value) else "missing")
    return sort_visibility_frame(status_df, ["scenario", "backbone", "scope", "mode"])


def format_float(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.3f}"


def describe_scope_effect(scope: str, effect_df: pd.DataFrame) -> str:
    frame = effect_df[(effect_df["scope"] == scope) & (effect_df["mode"] != "original")]
    if frame.empty:
        return f"{SCOPE_LABELS[scope]}：还没有可配对的完成结果。"
    mean_delta_ood = frame["delta_ood_accuracy"].mean()
    mean_delta_gap = frame["delta_normalized_gap"].mean()
    positive_count = int((frame["delta_ood_accuracy"] > 0).sum())
    total = int(frame["delta_ood_accuracy"].notna().sum())
    best = frame.sort_values(["delta_ood_accuracy", "delta_normalized_gap"], ascending=[False, True]).iloc[0]
    return (
        f"{SCOPE_LABELS[scope]}：平均 OOD accuracy 变化为 {format_float(mean_delta_ood)}，"
        f"平均 normalized gap 变化为 {format_float(mean_delta_gap)}，OOD accuracy 为正的配对结果为 {positive_count}/{total}。"
        f"当前最好的配对结果是 {best['scenario']} / {best['backbone']} / {MODE_LABELS[str(best['mode'])]}，"
        f"delta OOD 为 {format_float(best['delta_ood_accuracy'])}。"
    )


def build_interpretation_lines(
    metrics_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    split_effect_df: pd.DataFrame,
) -> list[str]:
    if metrics_df.empty:
        return [
            "当前没有发现已完成的 visibility runs，因此还不能解释 visibility hypothesis。",
            "请先根据 experiment_matrix.csv 和 missing_runs.csv 提交或收集完整结果；在指标出现前不要声称有提升。",
        ]

    lines = [
        describe_scope_effect("test_only", effect_df),
        describe_scope_effect("train_test_consistent", effect_df),
        describe_scope_effect("night_only", effect_df),
    ]

    non_original = effect_df[effect_df["mode"] != "original"].copy()
    if not non_original.empty:
        paired = non_original.pivot_table(
            index=["scenario", "backbone", "mode"],
            columns="scope",
            values="delta_ood_accuracy",
            aggfunc="first",
            observed=True,
        ).reset_index()
        if {"test_only", "train_test_consistent"}.issubset(paired.columns):
            paired = paired.dropna(subset=["test_only", "train_test_consistent"])
            if paired.empty:
                lines.append("Train+test consistent 与 test-only 对比：还没有完整配对的结果。")
            else:
                diff = paired["train_test_consistent"] - paired["test_only"]
                lines.append(
                    "Train+test consistent 相比 test-only 的平均配对 OOD delta 差值为 "
                    f"{format_float(diff.mean())}；其中 {int((diff > 0).sum())}/{len(diff)} 个配对结果更好。"
                )

        scenario_effect = (
            non_original.groupby("scenario", observed=True)["delta_ood_accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        if not scenario_effect.empty:
            lines.append(
                "平均 OOD 收益最大的 scenario 是 "
                f"{scenario_effect.index[0]}（相对 original 为 {format_float(scenario_effect.iloc[0])}）。"
            )

    if not metrics_df.empty:
        hardest = metrics_df.groupby("scenario", observed=True)["ood_accuracy"].mean().sort_values()
        if not hardest.empty:
            lines.append(
                f"按平均 OOD accuracy 看，最困难的 scenario 是 {hardest.index[0]} "
                f"（平均 OOD accuracy {hardest.iloc[0]:.3f}）。"
            )

    day_rows = split_effect_df[
        (split_effect_df["scope"] == "night_only")
        & (split_effect_df["mode"] != "original")
        & (split_effect_df["split_light"] == "day")
    ]
    if day_rows.empty:
        lines.append("Night-only 对 daytime 的影响：还没有可配对的 daytime split 结果。")
    else:
        mean_day_delta = day_rows["delta_accuracy"].mean()
        harmful = int((day_rows["delta_accuracy"] < 0).sum())
        lines.append(
            f"Night-only 对 daytime 的影响：daytime split accuracy 平均变化为 {format_float(mean_day_delta)}；"
            f"{harmful}/{len(day_rows)} 个配对结果为负。"
        )

    test_only = non_original[non_original["scope"] == "test_only"] if not non_original.empty else pd.DataFrame()
    night_only = non_original[non_original["scope"] == "night_only"] if not non_original.empty else pd.DataFrame()
    if test_only.empty:
        lines.append("Visibility-loss 证据：不完整，因为 test-only enhancement 结果缺失。")
    elif test_only["delta_ood_accuracy"].mean() > 0 and test_only["delta_normalized_gap"].mean() < 0:
        lines.append(
            "Visibility-loss 证据：test-only 的 OOD accuracy 平均提升且 normalized gap 平均下降，支持低可见度是重要失败模式。"
        )
    elif not night_only.empty and night_only["delta_ood_accuracy"].mean() > 0:
        lines.append(
            "Visibility-loss 证据：混合；night-aware retraining 平均有帮助，但 test-only enhancement 本身还不足以解释全部失败。"
        )
    else:
        lines.append(
            "Visibility-loss 证据：较弱或混合；除非配对指标确实改善，否则不要声称 visibility 是主要失败原因。"
        )

    return lines


def write_chinese_summary(
    output_path: Path,
    metrics_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    split_effect_df: pd.DataFrame,
    matrix_status_df: pd.DataFrame,
) -> None:
    completed = int((matrix_status_df["status"] == "completed").sum()) if not matrix_status_df.empty else 0
    missing = int((matrix_status_df["status"] == "missing").sum()) if not matrix_status_df.empty else 0
    lines = [
        "# 可见度假设实验总结",
        "",
        f"- 预期运行数：{len(matrix_status_df)}",
        f"- 本次聚合中已完成：{completed}",
        f"- 缺失：{missing}",
        "",
        "## 结果解读",
        "",
    ]
    for line in build_interpretation_lines(metrics_df, effect_df, split_effect_df):
        lines.append(f"- {line}")
    lines.extend(
        [
            "",
            "## 需要明确回答的问题",
            "",
            "- 只在 test time 改善可见度是否有帮助？看上面的 Test-time only 行；只有当 delta OOD 为正且 delta normalized gap 为负时才回答“有帮助”。",
            "- train+test consistent enhancement 是否比 test-only 更有帮助？看上面的配对差值。",
            "- night-only enhancement 是否在不伤害 daytime performance 的情况下有帮助？需要同时看 night-only 的 OOD 效果和 daytime split delta。",
            "- 哪个 scenario 受益最多？看平均 OOD 收益最大的 scenario。",
            "- 哪个 scenario 仍然最难？看平均 OOD accuracy 最低的 scenario。",
            "- 证据是否支持 visibility loss 是主要失败模式？看 visibility-loss 证据行；结论必须以实测指标为准。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_path: Path, output_dir: Path, matrix_status_df: pd.DataFrame) -> None:
    completed = int((matrix_status_df["status"] == "completed").sum()) if not matrix_status_df.empty else 0
    lines = [
        "# Visibility Experiment Run Manifest",
        "",
        "## Matrix",
        "",
        "- Scopes: test_only, train_test_consistent, night_only",
        "- Modes: original, gamma, clahe, gamma_clahe",
        "- Scenarios: cross_location, day_to_night, night_to_day",
        "- Backbones: resnet18, resnet34, resnet50, resnet101",
        "- Expected combinations: 144",
        f"- Completed combinations in this aggregation: {completed}/144",
        "",
        "## Output Files",
        "",
        f"- Output directory: `{output_dir}`",
        "- `visibility_runs.csv`: one row per selected run directory",
        "- `visibility_split_metrics.csv`: one row per evaluated split",
        "- `visibility_summary.csv`: tidy in-domain/OOD/gap table",
        "- `visibility_effect_metrics.csv`: deltas against original within each scope",
        "- `visibility_split_effect_metrics.csv`: split-level deltas against original",
        "- `experiment_matrix.csv`: expected matrix with completion status",
        "- `missing_runs.csv`: runs not found in the results root",
        "- `visibility_summary_zh.md`: interpretation summary",
        "- `sanity_checks.json`: aggregation-level counts and status",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_sanity_checks(matrix_status_df: pd.DataFrame, runs_df: pd.DataFrame) -> dict[str, Any]:
    completed = int((matrix_status_df["status"] == "completed").sum()) if not matrix_status_df.empty else 0
    missing = int((matrix_status_df["status"] == "missing").sum()) if not matrix_status_df.empty else 0
    eval_only_ok: bool | str
    if runs_df.empty:
        eval_only_ok = "no_completed_runs_to_verify"
    else:
        test_only_runs = runs_df[runs_df["scope"] == "test_only"]
        eval_only_ok = bool(test_only_runs["eval_only"].fillna(False).astype(bool).all()) if not test_only_runs.empty else True
    return {
        "expected_combination_count": int(len(matrix_status_df)),
        "completed_combination_count": completed,
        "missing_combination_count": missing,
        "all_expected_combinations_present": missing == 0,
        "test_only_runs_are_eval_only": eval_only_ok,
        "run_rows_seen": int(len(runs_df)),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_summary_paths = find_summary_files(results_root)
    summary_paths = select_latest_summary_paths(discovered_summary_paths)
    runs_df, splits_df = build_rows(summary_paths)
    metrics_df = build_visibility_metrics(splits_df)
    effect_df = build_effect_metrics(metrics_df)
    split_effect_df = build_split_effect_metrics(splits_df)
    expected_df = build_expected_matrix()
    matrix_status_df = build_matrix_status(expected_df, metrics_df)
    missing_runs_df = matrix_status_df[matrix_status_df["status"] == "missing"].copy()

    runs_df.to_csv(output_dir / "visibility_runs.csv", index=False)
    splits_df.to_csv(output_dir / "visibility_split_metrics.csv", index=False)
    metrics_df.to_csv(output_dir / "visibility_summary.csv", index=False)
    metrics_df.to_csv(output_dir / "visibility_metrics.csv", index=False)
    effect_df.to_csv(output_dir / "visibility_effect_metrics.csv", index=False)
    split_effect_df.to_csv(output_dir / "visibility_split_effect_metrics.csv", index=False)
    matrix_status_df.to_csv(output_dir / "experiment_matrix.csv", index=False)
    missing_runs_df.to_csv(output_dir / "missing_runs.csv", index=False)
    write_chinese_summary(output_dir / "visibility_summary_zh.md", metrics_df, effect_df, split_effect_df, matrix_status_df)
    write_manifest(output_dir / "RUN_MANIFEST.md", output_dir, matrix_status_df)
    (output_dir / "sanity_checks.json").write_text(
        json.dumps(build_sanity_checks(matrix_status_df, runs_df), indent=2),
        encoding="utf-8",
    )

    concise_table = metrics_df[
        ["scenario", "backbone", "scope", "mode", "in_domain_accuracy", "ood_accuracy", "normalized_gap"]
    ].head(24) if not metrics_df.empty else pd.DataFrame()
    (output_dir / "visibility_comparison.md").write_text(
        "\n".join(
            [
                "# Visibility Comparison",
                "",
                dataframe_to_markdown(concise_table) if not concise_table.empty else "No completed visibility metrics were found.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    payload = {
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "discovered_summary_file_count": len(discovered_summary_paths),
        "selected_visibility_summary_file_count": len(summary_paths),
        "expected_combination_count": len(expected_df),
        "completed_combination_count": int((matrix_status_df["status"] == "completed").sum()),
        "missing_combination_count": int((matrix_status_df["status"] == "missing").sum()),
        "tidy_summary_csv": str(output_dir / "visibility_summary.csv"),
        "chinese_summary": str(output_dir / "visibility_summary_zh.md"),
        "missing_runs_csv": str(output_dir / "missing_runs.csv"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
