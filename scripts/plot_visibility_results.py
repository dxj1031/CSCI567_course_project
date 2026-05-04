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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots for visibility hypothesis experiments.")
    parser.add_argument(
        "--comparison-dir",
        required=True,
        help="Directory containing visibility_summary.csv and visibility_effect_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where visibility plot PNG files will be written.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    mapped_mode = frame["mode"].map(MODE_LABELS)
    frame["mode_label"] = mapped_mode.fillna(frame["mode_label"]) if "mode_label" in frame else mapped_mode
    mapped_scope = frame["scope"].map(SCOPE_LABELS)
    frame["scope_label"] = mapped_scope.fillna(frame["scope_label"]) if "scope_label" in frame else mapped_scope
    frame["scenario"] = pd.Categorical(frame["scenario"], categories=SCENARIO_ORDER, ordered=True)
    frame["backbone"] = pd.Categorical(frame["backbone"], categories=BACKBONE_ORDER, ordered=True)
    frame["scope"] = pd.Categorical(frame["scope"], categories=SCOPE_ORDER, ordered=True)
    frame["mode"] = pd.Categorical(frame["mode"], categories=MODE_ORDER, ordered=True)
    return frame.sort_values(["scenario", "backbone", "scope", "mode"]).reset_index(drop=True)


def load_effects(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    mapped_mode = frame["mode"].map(MODE_LABELS)
    frame["mode_label"] = mapped_mode.fillna(frame["mode_label"]) if "mode_label" in frame else mapped_mode
    frame["scenario"] = pd.Categorical(frame["scenario"], categories=SCENARIO_ORDER, ordered=True)
    frame["backbone"] = pd.Categorical(frame["backbone"], categories=BACKBONE_ORDER, ordered=True)
    frame["scope"] = pd.Categorical(frame["scope"], categories=SCOPE_ORDER, ordered=True)
    frame["mode"] = pd.Categorical(frame["mode"], categories=MODE_ORDER, ordered=True)
    return frame.sort_values(["scenario", "backbone", "scope", "mode"]).reset_index(drop=True)


def save_empty_plot(output_path: Path, title: str = "No Completed Visibility Runs") -> None:
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


def add_zero_lines(grid: sns.FacetGrid) -> None:
    for ax in grid.axes.flat:
        ax.axhline(0.0, color="black", linewidth=1.0)


def save_ood_accuracy_by_mode(metrics_df: pd.DataFrame, output_path: Path) -> None:
    grid = sns.catplot(
        data=metrics_df,
        kind="bar",
        row="scenario",
        col="scope_label",
        x="mode_label",
        y="ood_accuracy",
        hue="backbone",
        order=[MODE_LABELS[mode] for mode in MODE_ORDER],
        row_order=SCENARIO_ORDER,
        col_order=[SCOPE_LABELS[scope] for scope in SCOPE_ORDER],
        height=3.4,
        aspect=1.25,
        sharey=True,
    )
    grid.set_axis_labels("Visibility Mode", "OOD Accuracy")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0.0, min(1.0, metrics_df["ood_accuracy"].max() + 0.12))
    grid.figure.savefig(output_path, bbox_inches="tight")
    plt.close(grid.figure)


def save_normalized_gap_by_mode(metrics_df: pd.DataFrame, output_path: Path) -> None:
    grid = sns.catplot(
        data=metrics_df,
        kind="bar",
        row="scenario",
        col="scope_label",
        x="mode_label",
        y="normalized_gap",
        hue="backbone",
        order=[MODE_LABELS[mode] for mode in MODE_ORDER],
        row_order=SCENARIO_ORDER,
        col_order=[SCOPE_LABELS[scope] for scope in SCOPE_ORDER],
        height=3.4,
        aspect=1.25,
        sharey=True,
    )
    grid.set_axis_labels("Visibility Mode", "Normalized Gap")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0.0, max(0.05, min(1.0, metrics_df["normalized_gap"].max() + 0.12)))
    grid.figure.savefig(output_path, bbox_inches="tight")
    plt.close(grid.figure)


def save_delta_metric_grid(
    effect_df: pd.DataFrame,
    output_path: Path,
    metric: str,
    ylabel: str,
) -> None:
    frame = effect_df[effect_df["mode"] != "original"].copy()
    if frame.empty:
        raise ValueError("No non-original effect rows are available.")
    grid = sns.catplot(
        data=frame,
        kind="bar",
        row="scenario",
        col="scope",
        x="mode_label",
        y=metric,
        hue="backbone",
        order=[MODE_LABELS[mode] for mode in MODE_ORDER if mode != "original"],
        row_order=SCENARIO_ORDER,
        col_order=SCOPE_ORDER,
        height=3.4,
        aspect=1.25,
        sharey=True,
    )
    grid.set_axis_labels("Visibility Mode", ylabel)
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    add_zero_lines(grid)
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    grid.figure.savefig(output_path, bbox_inches="tight")
    plt.close(grid.figure)


def save_tradeoff_scatter(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    sns.scatterplot(
        data=metrics_df,
        x="normalized_gap",
        y="ood_accuracy",
        hue="mode_label",
        style="scope_label",
        size="depth",
        sizes=(80, 260),
        ax=ax,
    )
    for _, row in metrics_df.iterrows():
        ax.text(
            row["normalized_gap"] + 0.003,
            row["ood_accuracy"] + 0.003,
            f"{row['scenario']} {row['backbone']}",
            fontsize=8,
        )
    ax.set_title("Visibility Trade-off: Normalized Gap vs OOD Accuracy")
    ax.set_xlabel("Normalized Gap")
    ax.set_ylabel("OOD Accuracy")
    ax.legend(title="Mode / Scope / Depth", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_best_mode_heatmap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(SCOPE_ORDER), figsize=(18, 6), constrained_layout=True)
    if len(SCOPE_ORDER) == 1:
        axes = [axes]

    for ax, scope in zip(axes, SCOPE_ORDER):
        frame = metrics_df[metrics_df["scope"].astype(str) == scope].copy()
        best_rows = []
        for (scenario, backbone), group in frame.groupby(["scenario", "backbone"], observed=True):
            if group.empty:
                continue
            best = group.sort_values("ood_accuracy", ascending=False).iloc[0]
            best_rows.append(
                {
                    "scenario": scenario,
                    "backbone": backbone,
                    "ood_accuracy": best["ood_accuracy"],
                    "annotation": MODE_LABELS[str(best["mode"])],
                }
            )
        best_df = pd.DataFrame(best_rows)
        if best_df.empty:
            ax.set_axis_off()
            ax.set_title(SCOPE_LABELS[scope])
            continue
        values = best_df.pivot(index="scenario", columns="backbone", values="ood_accuracy").reindex(
            index=SCENARIO_ORDER,
            columns=BACKBONE_ORDER,
        )
        annotations = best_df.pivot(index="scenario", columns="backbone", values="annotation").reindex(
            index=SCENARIO_ORDER,
            columns=BACKBONE_ORDER,
        )
        sns.heatmap(
            values,
            annot=annotations,
            fmt="",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar=scope == SCOPE_ORDER[-1],
            ax=ax,
        )
        ax.set_title(SCOPE_LABELS[scope])
        ax.set_xlabel("Backbone")
        ax.set_ylabel("Scenario")

    fig.suptitle("Best Visibility Mode By Scenario And Backbone")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_scenario_plot(metrics_df: pd.DataFrame, scenario: str, output_path: Path) -> None:
    frame = metrics_df[metrics_df["scenario"].astype(str) == scenario].copy()
    if frame.empty:
        save_empty_plot(output_path, title=f"No Completed Runs: {scenario}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    specs = [
        ("ood_accuracy", "OOD Accuracy", "OOD Accuracy"),
        ("normalized_gap", "Normalized Gap", "Normalized Gap"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specs):
        sns.lineplot(
            data=frame,
            x="mode_label",
            y=metric,
            hue="scope_label",
            style="backbone",
            marker="o",
            linewidth=2.0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Visibility Mode")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
    axes[0].legend(title="Scope / Backbone", frameon=True)
    legend = axes[1].get_legend()
    if legend is not None:
        legend.remove()
    fig.suptitle(scenario.replace("_", " ").title())
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics(comparison_dir / "visibility_summary.csv")
    effect_df = load_effects(comparison_dir / "visibility_effect_metrics.csv")

    plot_paths = [
        output_dir / "visibility_ood_accuracy_by_mode.png",
        output_dir / "visibility_normalized_gap_by_mode.png",
        output_dir / "visibility_delta_ood_accuracy_vs_original.png",
        output_dir / "visibility_delta_normalized_gap_vs_original.png",
        output_dir / "visibility_tradeoff_scatter.png",
        output_dir / "visibility_best_mode_heatmap.png",
    ]
    scenario_paths = [
        output_dir / f"visibility_{scenario}_scenario_summary.png"
        for scenario in SCENARIO_ORDER
    ]

    setup_plot_style()
    if metrics_df.empty:
        for path in plot_paths + scenario_paths:
            save_empty_plot(path)
        payload = {
            "comparison_dir": str(comparison_dir),
            "output_dir": str(output_dir),
            "plots": [str(path) for path in plot_paths + scenario_paths],
            "notes": "visibility_summary.csv was empty; placeholder no-results plots were written.",
        }
        print(json.dumps(payload, indent=2))
        return

    if plt is None or sns is None:
        raise ModuleNotFoundError("matplotlib and seaborn are required to plot non-empty visibility metrics.")

    save_ood_accuracy_by_mode(metrics_df, plot_paths[0])
    save_normalized_gap_by_mode(metrics_df, plot_paths[1])
    if effect_df.empty:
        save_empty_plot(plot_paths[2], title="No Paired Effect Metrics")
        save_empty_plot(plot_paths[3], title="No Paired Effect Metrics")
    else:
        save_delta_metric_grid(
            effect_df,
            plot_paths[2],
            metric="delta_ood_accuracy",
            ylabel="Delta OOD Accuracy vs Original",
        )
        save_delta_metric_grid(
            effect_df,
            plot_paths[3],
            metric="delta_normalized_gap",
            ylabel="Delta Normalized Gap vs Original",
        )
    save_tradeoff_scatter(metrics_df, plot_paths[4])
    save_best_mode_heatmap(metrics_df, plot_paths[5])
    for scenario, path in zip(SCENARIO_ORDER, scenario_paths):
        save_scenario_plot(metrics_df, scenario, path)

    payload = {
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "plots": [str(path) for path in plot_paths + scenario_paths],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
