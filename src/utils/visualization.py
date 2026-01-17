"""
Visualization utilities for benchmark results.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import MODELS


def setup_style():
    """Set up matplotlib style for consistent plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def plot_comparison(
    results: pd.DataFrame,
    metric: str,
    title: str | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create a bar chart comparing models on a specific metric.

    Args:
        results: DataFrame with model results
        metric: Metric column to plot
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    models = results["model"].tolist()
    values = results[metric].tolist()
    colors = [MODELS.get(m.lower().replace("-", ""), {}).get("color", "#95a5a6") for m in models]

    bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{value:.4f}" if isinstance(value, float) else str(value),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title or f"{metric} Comparison", fontsize=14, fontweight="bold")

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    return fig


def plot_size_metrics(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create a grouped bar chart for AP by object size.

    Args:
        results: DataFrame with AP_small, AP_medium, AP_large columns
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    models = results["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    metrics = ["AP_small", "AP_medium", "AP_large"]
    labels = ["Small (<32px)", "Medium (32-96px)", "Large (>96px)"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = results[metric].tolist()
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="black")

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Average Precision (AP)", fontsize=12)
    ax.set_title("AP by Object Size (COCO Evaluation)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(title="Object Size", loc="upper right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    return fig


def plot_speed_comparison(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create a speed comparison chart.

    Args:
        results: DataFrame with speed benchmark results
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Group by device
    devices = results["device"].unique()

    for idx, device in enumerate(devices):
        ax = axes[idx]
        device_data = results[results["device"] == device]

        models = device_data["model"].tolist()
        times = device_data["mean_ms"].tolist()
        colors = [MODELS.get(m.lower().replace("-", ""), {}).get("color", "#95a5a6") for m in models]

        bars = ax.bar(models, times, color=colors, edgecolor="black", linewidth=1.2)

        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.annotate(
                f"{time:.1f}ms",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Inference Time (ms)", fontsize=12)
        ax.set_title(f"Inference Speed on {device.upper()}", fontsize=14, fontweight="bold")
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    return fig


def create_benchmark_report(
    metrics_df: pd.DataFrame,
    size_df: pd.DataFrame,
    speed_df: pd.DataFrame,
    output_dir: str | Path = "assets",
) -> dict[str, Path]:
    """
    Create a complete benchmark visualization report.

    Args:
        metrics_df: DataFrame with overall metrics
        size_df: DataFrame with size-based metrics
        speed_df: DataFrame with speed benchmark results
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = {}

    # mAP comparison
    if "mAP50-95" in metrics_df.columns:
        plot_comparison(
            metrics_df,
            "mAP50-95",
            title="mAP50-95 Comparison on VisDrone",
            output_path=output_dir / "map_comparison.png",
        )
        plots["map_comparison"] = output_dir / "map_comparison.png"

    if "mAP50" in metrics_df.columns:
        plot_comparison(
            metrics_df,
            "mAP50",
            title="mAP50 Comparison on VisDrone",
            output_path=output_dir / "map50_comparison.png",
        )
        plots["map50_comparison"] = output_dir / "map50_comparison.png"

    # Size-based metrics
    if all(col in size_df.columns for col in ["AP_small", "AP_medium", "AP_large"]):
        plot_size_metrics(
            size_df,
            output_path=output_dir / "map_by_size.png",
        )
        plots["map_by_size"] = output_dir / "map_by_size.png"

    # Speed comparison
    if "mean_ms" in speed_df.columns:
        plot_speed_comparison(
            speed_df,
            output_path=output_dir / "speed_comparison.png",
        )
        plots["speed_comparison"] = output_dir / "speed_comparison.png"

    print(f"\nGenerated {len(plots)} visualization plots in {output_dir}")
    return plots


def main():
    """CLI entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load available results
    metrics_path = results_dir / "metrics.csv"
    size_path = results_dir / "coco_eval.csv"
    speed_path = results_dir / "speed_benchmark.csv"

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = pd.DataFrame()

    if size_path.exists():
        size_df = pd.read_csv(size_path)
    else:
        size_df = pd.DataFrame()

    if speed_path.exists():
        speed_df = pd.read_csv(speed_path)
    else:
        speed_df = pd.DataFrame()

    if metrics_df.empty and size_df.empty and speed_df.empty:
        print("No result files found. Run benchmarks first.")
        return

    create_benchmark_report(
        metrics_df=metrics_df,
        size_df=size_df,
        speed_df=speed_df,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
