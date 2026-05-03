#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_capacity_results import setup_plot_style


VARIANT_ORDER = ["original", "bbox_bg", "histmatch"]
VARIANT_LABELS = {
    "original": "Original",
    "bbox_bg": "BBox Blur",
    "histmatch": "Brightness Aligned",
}
SCENARIO_ORDER = ["cross_location", "day_to_night", "night_to_day"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create scatter and bar plots for ResNet50 dataset-intervention experiments."
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
    return frame.sort_values(["scenario", "variant"]).reset_index(drop=True)


def build_effect_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_columns = [
        "in_domain_accuracy",
        "out_of_domain_accuracy",
        "gap_accuracy",
        "normalized_gap",
    ]

    for scenario, frame in metrics_df.groupby("scenario", observed=True, sort=False):
        original = frame[frame["variant"] == "original"]
        if original.empty:
            continue
        baseline = original.iloc[0]

        for _, row in frame.iterrows():
            effect_row: dict[str, object] = {
                "scenario": scenario,
                "variant": row["variant"],
                "variant_label": row["variant_label"],
            }
            for metric in metric_columns:
                effect_row[f"original_{metric}"] = baseline[metric]
                effect_row[metric] = row[metric]
                effect_row[f"delta_{metric}"] = row[metric] - baseline[metric]
            rows.append(effect_row)

    if not rows:
        columns = ["scenario", "variant", "variant_label"]
        for metric in metric_columns:
            columns.extend([f"original_{metric}", metric, f"delta_{metric}"])
        return pd.DataFrame(columns=columns)

    effect_df = pd.DataFrame(rows)
    effect_df["scenario"] = pd.Categorical(effect_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    effect_df["variant"] = pd.Categorical(effect_df["variant"], categories=VARIANT_ORDER, ordered=True)
    return effect_df.sort_values(["scenario", "variant"]).reset_index(drop=True)


def save_scatter(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.scatterplot(
        data=metrics_df,
        x="normalized_gap",
        y="out_of_domain_accuracy",
        hue="scenario",
        style="variant_label",
        s=180,
        ax=ax,
    )

    for _, row in metrics_df.iterrows():
        ax.text(
            row["normalized_gap"] + 0.004,
            row["out_of_domain_accuracy"] + 0.004,
            row["variant_label"],
            fontsize=10,
        )

    ax.set_title("ResNet50 Interventions: Normalized Gap vs OOD Accuracy")
    ax.set_xlabel("Normalized Gap (gap / in-domain accuracy)")
    ax.set_ylabel("Out-of-Domain Accuracy")
    ax.legend(title="Scenario / Variant", frameon=True)
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


def save_bar_grid(metrics_df: pd.DataFrame, output_path: Path) -> None:
    scenarios = metrics_df["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6), constrained_layout=True)

    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        frame = metrics_df[metrics_df["scenario"] == scenario].copy()
        long_frame = frame.melt(
            id_vars=["scenario", "variant", "variant_label"],
            value_vars=["in_domain_accuracy", "out_of_domain_accuracy"],
            var_name="domain_type",
            value_name="accuracy",
        )
        long_frame["domain_type"] = long_frame["domain_type"].map(
            {
                "in_domain_accuracy": "In-domain",
                "out_of_domain_accuracy": "Out-of-domain",
            }
        )

        sns.barplot(
            data=long_frame,
            x="variant_label",
            y="accuracy",
            hue="domain_type",
            ax=ax,
        )
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_xlabel("Dataset Variant")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, min(1.0, long_frame["accuracy"].max() + 0.15))
        ax.tick_params(axis="x", rotation=25)

    axes[0].legend(title="Evaluation Domain", frameon=True)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_variant_trend_lines(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    sns.lineplot(
        data=metrics_df,
        x="variant_label",
        y="out_of_domain_accuracy",
        hue="scenario",
        marker="o",
        linewidth=2.5,
        ax=axes[0],
    )
    axes[0].set_title("OOD Accuracy By Dataset Variant")
    axes[0].set_xlabel("Training Dataset Variant")
    axes[0].set_ylabel("Out-of-Domain Accuracy")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylim(0.0, min(1.0, metrics_df["out_of_domain_accuracy"].max() + 0.1))

    sns.lineplot(
        data=metrics_df,
        x="variant_label",
        y="normalized_gap",
        hue="scenario",
        marker="o",
        linewidth=2.5,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Normalized Gap By Dataset Variant")
    axes[1].set_xlabel("Training Dataset Variant")
    axes[1].set_ylabel("Normalized Gap (gap / in-domain accuracy)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylim(0.0, min(1.0, metrics_df["normalized_gap"].max() + 0.1))

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=labels, title="Scenario", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_effect_delta_grid(effect_df: pd.DataFrame, output_path: Path) -> None:
    intervention_effects = effect_df[effect_df["variant"] != "original"].copy()
    if intervention_effects.empty:
        raise ValueError("No non-original intervention rows were available for delta plots.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    plot_specs = [
        (
            "delta_out_of_domain_accuracy",
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
            x="scenario",
            y=metric,
            hue="variant_label",
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Scenario")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        add_bar_labels(ax)

    axes[0].legend(title="Intervention", frameon=True)
    legend = axes[1].get_legend()
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
    if metrics_df.empty:
        raise ValueError("intervention_metrics.csv was empty; there is nothing to plot.")

    effect_df = build_effect_metrics(metrics_df)
    effect_metrics_path = output_dir / "intervention_effect_metrics.csv"
    effect_df.to_csv(effect_metrics_path, index=False)

    setup_plot_style()
    scatter_path = output_dir / "intervention_tradeoff_scatter.png"
    bar_grid_path = output_dir / "intervention_in_out_bar_grid.png"
    trend_lines_path = output_dir / "intervention_variant_trend_lines.png"
    delta_grid_path = output_dir / "intervention_delta_bar_grid.png"

    save_scatter(metrics_df, scatter_path)
    save_bar_grid(metrics_df, bar_grid_path)
    save_variant_trend_lines(metrics_df, trend_lines_path)
    save_effect_delta_grid(effect_df, delta_grid_path)

    payload = {
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "effect_metrics_csv": str(effect_metrics_path),
        "plots": [
            str(scatter_path),
            str(bar_grid_path),
            str(trend_lines_path),
            str(delta_grid_path),
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
