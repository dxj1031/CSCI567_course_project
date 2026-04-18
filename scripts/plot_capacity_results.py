#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create intuitive plots from capacity comparison CSV outputs."
    )
    parser.add_argument(
        "--comparison-dir",
        required=True,
        help="Directory containing capacity_trend.csv and related comparison files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plot PNG files will be written.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200


def add_normalized_gap(trend_df: pd.DataFrame) -> pd.DataFrame:
    frame = trend_df.copy()
    frame["normalized_gap"] = frame["drop_accuracy"] / frame["in_domain_accuracy"]
    return frame


def save_capacity_trend_lines(trend_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    sns.lineplot(
        data=trend_df,
        x="depth",
        y="out_of_domain_accuracy",
        hue="scenario",
        marker="o",
        linewidth=2.5,
        ax=axes[0],
    )
    axes[0].set_title("OOD Accuracy vs Model Depth")
    axes[0].set_xlabel("ResNet Depth")
    axes[0].set_ylabel("Out-of-Domain Accuracy")
    axes[0].set_xticks(sorted(trend_df["depth"].dropna().unique()))
    axes[0].set_ylim(0.0, min(1.0, trend_df["out_of_domain_accuracy"].max() + 0.1))

    sns.lineplot(
        data=trend_df,
        x="depth",
        y="normalized_gap",
        hue="scenario",
        marker="o",
        linewidth=2.5,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Normalized Gap vs Model Depth")
    axes[1].set_xlabel("ResNet Depth")
    axes[1].set_ylabel("Normalized Gap (gap / in-domain accuracy)")
    axes[1].set_xticks(sorted(trend_df["depth"].dropna().unique()))
    axes[1].set_ylim(0.0, min(1.0, trend_df["normalized_gap"].max() + 0.1))

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=labels, title="Scenario", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_capacity_tradeoff_scatter(trend_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    sns.scatterplot(
        data=trend_df,
        x="normalized_gap",
        y="out_of_domain_accuracy",
        hue="scenario",
        size="depth",
        sizes=(80, 260),
        ax=ax,
    )

    for _, row in trend_df.iterrows():
        ax.text(
            row["normalized_gap"] + 0.003,
            row["out_of_domain_accuracy"] + 0.003,
            row["backbone"],
            fontsize=10,
        )

    ax.set_title("Capacity Trade-off: Normalized Gap vs OOD Accuracy")
    ax.set_xlabel("Normalized Gap (gap / in-domain accuracy)")
    ax.set_ylabel("Out-of-Domain Accuracy")
    ax.legend(title="Scenario / Depth", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_in_out_bar_grid(trend_df: pd.DataFrame, output_path: Path) -> None:
    scenarios = trend_df["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6), constrained_layout=True)

    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        frame = trend_df[trend_df["scenario"] == scenario].copy()
        frame = frame.sort_values(["depth", "backbone"])
        long_frame = frame.melt(
            id_vars=["scenario", "backbone", "depth"],
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
            x="backbone",
            y="accuracy",
            hue="domain_type",
            ax=ax,
        )
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_xlabel("Backbone")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, min(1.0, long_frame["accuracy"].max() + 0.15))
        ax.tick_params(axis="x", rotation=30)

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

    trend_df = load_csv(comparison_dir / "capacity_trend.csv").sort_values(
        ["scenario", "depth", "backbone"],
        na_position="last",
    )
    trend_df = add_normalized_gap(trend_df)

    setup_plot_style()
    line_plot_path = output_dir / "capacity_trend_lines.png"
    scatter_plot_path = output_dir / "capacity_tradeoff_scatter.png"
    bar_grid_path = output_dir / "capacity_in_out_bar_grid.png"

    save_capacity_trend_lines(trend_df, line_plot_path)
    save_capacity_tradeoff_scatter(trend_df, scatter_plot_path)
    save_in_out_bar_grid(trend_df, bar_grid_path)

    payload = {
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "plots": [
            str(line_plot_path),
            str(scatter_plot_path),
            str(bar_grid_path),
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
