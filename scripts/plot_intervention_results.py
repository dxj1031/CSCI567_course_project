#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from plot_capacity_results import setup_plot_style
except ModuleNotFoundError:
    plt = None
    sns = None

    def setup_plot_style() -> None:
        return None


VARIANT_LABELS = {
    "original": "Original",
    "photometric_randomization": "Photometric Randomization",
    "background_perturbation": "Background Perturbation",
    "combined": "Combined",
}
SCENARIO_ORDER = ["cross_location", "day_to_night", "night_to_day"]
VARIANT_ORDER = ["original", "photometric_randomization", "background_perturbation", "combined"]
BACKBONE_ORDER = ["resnet18", "resnet34", "resnet50", "resnet101"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots for train-time intervention experiments."
    )
    parser.add_argument(
        "--comparison-dir",
        required=True,
        help="Directory containing intervention_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where intervention plot PNG files will be written.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["variant_label"] = frame["variant"].map(VARIANT_LABELS).fillna(frame["variant_label"])
    frame["variant"] = pd.Categorical(frame["variant"], categories=VARIANT_ORDER, ordered=True)
    frame["scenario"] = pd.Categorical(frame["scenario"], categories=SCENARIO_ORDER, ordered=True)
    frame["backbone"] = pd.Categorical(frame["backbone"], categories=BACKBONE_ORDER, ordered=True)
    return frame.sort_values(["scenario", "backbone", "variant"]).reset_index(drop=True)


def build_effect_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_columns = [
        "in_domain_accuracy",
        "out_of_domain_accuracy",
        "gap_accuracy",
        "in_domain_acc",
        "ood_acc",
        "gap",
        "normalized_gap",
    ]
    metric_columns = [column for column in metric_columns if column in metrics_df.columns]
    metadata_columns = [column for column in ["seed", "run_id", "timestamp", "notes"] if column in metrics_df.columns]

    for (scenario, backbone), frame in metrics_df.groupby(["scenario", "backbone"], observed=True, sort=False):
        original = frame[frame["variant"] == "original"]
        if original.empty:
            continue
        baseline = original.iloc[0]

        for _, row in frame.iterrows():
            effect_row: dict[str, object] = {
                "scenario": scenario,
                "backbone": backbone,
                "variant": row["variant"],
                "variant_label": row["variant_label"],
            }
            for column in metadata_columns:
                effect_row[column] = row[column]
            for metric in metric_columns:
                effect_row[f"original_{metric}"] = baseline[metric]
                effect_row[metric] = row[metric]
                effect_row[f"delta_{metric}"] = row[metric] - baseline[metric]
            rows.append(effect_row)

    if not rows:
        columns = ["scenario", "backbone", "variant", "variant_label"] + metadata_columns
        for metric in metric_columns:
            columns.extend([f"original_{metric}", metric, f"delta_{metric}"])
        return pd.DataFrame(columns=columns)

    effect_df = pd.DataFrame(rows)
    effect_df["scenario"] = pd.Categorical(effect_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    effect_df["backbone"] = pd.Categorical(effect_df["backbone"], categories=BACKBONE_ORDER, ordered=True)
    effect_df["variant"] = pd.Categorical(effect_df["variant"], categories=VARIANT_ORDER, ordered=True)
    return effect_df.sort_values(["scenario", "backbone", "variant"]).reset_index(drop=True)


def save_scatter(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.scatterplot(
        data=metrics_df,
        x="normalized_gap",
        y="ood_acc",
        hue="variant_label",
        style="scenario",
        size="backbone",
        sizes=(80, 260),
        ax=ax,
    )

    for _, row in metrics_df.iterrows():
        ax.text(
            row["normalized_gap"] + 0.004,
            row["ood_acc"] + 0.004,
            row["backbone"],
            fontsize=10,
        )

    ax.set_title("Train-Time Interventions: Normalized Gap vs OOD Accuracy")
    ax.set_xlabel("Normalized Gap (gap / in-domain accuracy)")
    ax.set_ylabel("Out-of-Domain Accuracy")
    ax.legend(title="Variant / Scenario / Backbone", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, fmt: str = "{:+.3f}") -> None:
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[fmt.format(value) if pd.notna(value) else "" for value in container.datavalues],
            fontsize=9,
            padding=3,
        )


def save_empty_plot(output_path: Path, title: str = "No Completed Runs") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        image = Image.new("RGB", (1400, 900), "white")
        draw = ImageDraw.Draw(image)
        draw.text((70, 70), title, fill="black")
        draw.text((70, 150), "No completed experiment metrics were found.", fill="black")
        image.save(output_path)
        return

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.text(
        0.5,
        0.5,
        "No completed experiment metrics were found.",
        ha="center",
        va="center",
        fontsize=16,
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_ood_accuracy_by_variant(metrics_df: pd.DataFrame, output_path: Path) -> None:
    scenarios = metrics_df["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6), constrained_layout=True)

    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        frame = metrics_df[metrics_df["scenario"] == scenario].copy()
        sns.barplot(
            data=frame,
            x="variant_label",
            y="ood_acc",
            hue="backbone",
            ax=ax,
        )
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_xlabel("Training Variant")
        ax.set_ylabel("Out-of-Domain Accuracy")
        ax.set_ylim(0.0, min(1.0, frame["ood_acc"].max() + 0.15))
        ax.tick_params(axis="x", rotation=20)

    axes[0].legend(title="Backbone", frameon=True)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metric_by_variant_lines(metrics_df: pd.DataFrame, output_path: Path, metric: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    sns.lineplot(
        data=metrics_df,
        x="variant_label",
        y=metric,
        hue="scenario",
        style="backbone",
        marker="o",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Training Variant")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0.0, min(1.0, metrics_df[metric].max() + 0.1))
    ax.legend(title="Scenario / Backbone", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_effect_delta_grid(effect_df: pd.DataFrame, output_path: Path) -> None:
    intervention_effects = effect_df[effect_df["variant"] != "original"].copy()
    if intervention_effects.empty:
        raise ValueError("No non-original intervention rows were available for delta plots.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    plot_specs = [
        (
            "delta_ood_acc",
            "Change In OOD Accuracy vs Original",
            "Delta OOD Accuracy (higher is better)",
        ),
        (
            "delta_normalized_gap",
            "Change In Normalized Gap vs Original",
            "Delta Normalized Gap (lower is better)",
        ),
    ]

    for ax, (metric, title, ylabel) in zip(axes, plot_specs):
        sns.barplot(
            data=intervention_effects,
            x="backbone",
            y=metric,
            hue="variant_label",
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Backbone")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        add_bar_labels(ax)

    axes[0].legend(title="Intervention", frameon=True)
    legend = axes[1].get_legend()
    if legend is not None:
        legend.remove()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_delta_metric_by_scenario(
    effect_df: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    intervention_effects = effect_df[effect_df["variant"] != "original"].copy()
    if intervention_effects.empty:
        raise ValueError("No non-original intervention rows were available for delta plots.")

    scenarios = intervention_effects["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6), constrained_layout=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        frame = intervention_effects[intervention_effects["scenario"] == scenario].copy()
        sns.barplot(
            data=frame,
            x="backbone",
            y=metric,
            hue="variant_label",
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(str(scenario).replace("_", " ").title())
        ax.set_xlabel("Backbone")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        add_bar_labels(ax)

    axes[0].legend(title="Intervention", frameon=True)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.suptitle(title)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_backbone_lines_by_scenario(metrics_df: pd.DataFrame, output_path: Path) -> None:
    scenarios = metrics_df["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6), constrained_layout=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        frame = metrics_df[metrics_df["scenario"] == scenario].copy()
        sns.lineplot(
            data=frame,
            x="backbone",
            y="ood_acc",
            hue="variant_label",
            marker="o",
            linewidth=2.5,
            ax=ax,
        )
        ax.set_title(str(scenario).replace("_", " ").title())
        ax.set_xlabel("Backbone")
        ax.set_ylabel("Out-of-Domain Accuracy")
        ax.set_ylim(0.0, min(1.0, frame["ood_acc"].max() + 0.15))
        ax.tick_params(axis="x", rotation=20)

    axes[0].legend(title="Training Variant", frameon=True)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_all_backbones_variants_grid(metrics_df: pd.DataFrame, output_path: Path) -> None:
    grid = sns.catplot(
        data=metrics_df,
        kind="bar",
        row="scenario",
        col="variant_label",
        x="backbone",
        y="ood_acc",
        order=BACKBONE_ORDER,
        col_order=[VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
        row_order=SCENARIO_ORDER,
        height=3.3,
        aspect=1.3,
        sharey=True,
    )
    grid.set_axis_labels("Backbone", "Out-of-Domain Accuracy")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0.0, min(1.0, metrics_df["ood_acc"].max() + 0.15))
    grid.figure.savefig(output_path, bbox_inches="tight")
    plt.close(grid.figure)


def save_backbone_comparison_grid(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    specs = [
        ("ood_acc", "OOD Accuracy", "Out-of-Domain Accuracy"),
        ("normalized_gap", "Normalized Gap", "Normalized Gap"),
        ("gap", "Absolute Gap", "Gap"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, specs):
        sns.lineplot(
            data=metrics_df,
            x="backbone",
            y=metric,
            hue="variant_label",
            style="scenario",
            marker="o",
            linewidth=2.0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Backbone")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)

    axes[0].legend(title="Variant / Scenario", frameon=True)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics(comparison_dir / "intervention_metrics.csv")
    effect_metrics_path = output_dir / "intervention_effect_metrics.csv"
    scatter_path = output_dir / "intervention_tradeoff_scatter.png"
    ood_by_variant_path = output_dir / "intervention_ood_accuracy_by_variant.png"
    gap_by_variant_path = output_dir / "intervention_normalized_gap_by_variant.png"
    delta_grid_path = output_dir / "intervention_delta_bar_grid.png"
    delta_ood_path = output_dir / "intervention_delta_ood_accuracy_vs_original.png"
    delta_gap_path = output_dir / "intervention_delta_normalized_gap_vs_original.png"
    backbone_lines_path = output_dir / "intervention_backbone_lines_by_scenario.png"
    backbone_grid_path = output_dir / "intervention_backbone_comparison.png"
    all_grid_path = output_dir / "intervention_all_backbones_variants_grid.png"
    plot_paths = [
        scatter_path,
        ood_by_variant_path,
        gap_by_variant_path,
        delta_grid_path,
        delta_ood_path,
        delta_gap_path,
        backbone_lines_path,
        backbone_grid_path,
        all_grid_path,
    ]

    effect_df = build_effect_metrics(metrics_df)
    effect_df.to_csv(effect_metrics_path, index=False)

    setup_plot_style()
    if metrics_df.empty:
        for plot_path in plot_paths:
            save_empty_plot(plot_path)
        payload = {
            "comparison_dir": str(comparison_dir),
            "output_dir": str(output_dir),
            "effect_metrics_csv": str(effect_metrics_path),
            "plots": [str(path) for path in plot_paths],
            "notes": "intervention_metrics.csv was empty; placeholder no-results plots were written.",
        }
        print(json.dumps(payload, indent=2))
        return

    if plt is None or sns is None:
        raise ModuleNotFoundError("matplotlib and seaborn are required to plot non-empty intervention metrics.")

    save_scatter(metrics_df, scatter_path)
    save_ood_accuracy_by_variant(metrics_df, ood_by_variant_path)
    save_metric_by_variant_lines(
        metrics_df,
        gap_by_variant_path,
        metric="normalized_gap",
        title="Normalized Gap By Training Variant",
        ylabel="Normalized Gap (gap / in-domain accuracy)",
    )
    save_effect_delta_grid(effect_df, delta_grid_path)
    save_delta_metric_by_scenario(
        effect_df,
        delta_ood_path,
        metric="delta_ood_acc",
        title="Delta OOD Accuracy vs Original",
        ylabel="Delta OOD Accuracy (higher is better)",
    )
    save_delta_metric_by_scenario(
        effect_df,
        delta_gap_path,
        metric="delta_normalized_gap",
        title="Delta Normalized Gap vs Original",
        ylabel="Delta Normalized Gap (lower is better)",
    )
    save_backbone_lines_by_scenario(metrics_df, backbone_lines_path)
    save_backbone_comparison_grid(metrics_df, backbone_grid_path)
    save_all_backbones_variants_grid(metrics_df, all_grid_path)

    payload = {
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "effect_metrics_csv": str(effect_metrics_path),
        "plots": [
            str(scatter_path),
            str(ood_by_variant_path),
            str(gap_by_variant_path),
            str(delta_grid_path),
            str(delta_ood_path),
            str(delta_gap_path),
            str(backbone_lines_path),
            str(backbone_grid_path),
            str(all_grid_path),
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
