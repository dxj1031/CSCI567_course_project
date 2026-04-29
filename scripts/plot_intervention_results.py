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
    frame["variant"] = pd.Categorical(frame["variant"], categories=VARIANT_ORDER, ordered=True)
    return frame.sort_values(["scenario", "variant"]).reset_index(drop=True)


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


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics(comparison_dir / "intervention_metrics.csv")
    if metrics_df.empty:
        raise ValueError("intervention_metrics.csv was empty; there is nothing to plot.")

    setup_plot_style()
    scatter_path = output_dir / "intervention_tradeoff_scatter.png"
    bar_grid_path = output_dir / "intervention_in_out_bar_grid.png"

    save_scatter(metrics_df, scatter_path)
    save_bar_grid(metrics_df, bar_grid_path)

    payload = {
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "plots": [
            str(scatter_path),
            str(bar_grid_path),
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
