#!/usr/bin/env python3
"""
Plot Sella TS Results

This script loads all Sella transition state search results from the plots directory
and creates comprehensive visualizations comparing RMSD and timing metrics across
different methods and starting points.

Usage:
    python playground/plot_sella_results.py
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple

# Set matplotlib style
plt.style.use("default")
sns.set_palette("husl")

# # Set global font size for all plots
# plt.rcParams.update({
#     'font.size': 4,
#     'axes.titlesize': 5,
#     'axes.labelsize': 4,
#     'xtick.labelsize': 3,
#     'ytick.labelsize': 3,
#     'legend.fontsize': 4,
#     'figure.titlesize': 6
# })


def parse_experiment_name(dirname: str) -> Dict[str, str]:
    """
    Parse experiment directory name to extract metadata.

    Example: 'sella_ts_from_linear_r_p_hessian_none_idx_0' ->
    {
        'method': 'sella_ts',
        'start_point': 'linear_r_p',
        'hessian': 'none',
        'coordinates': 'cartesian',
        'idx': '0'
    }
    """
    parts = dirname.split("_")

    metadata = {
        "method": "sella_ts",
        "start_point": "unknown",
        "hessian": "none",
        "coordinates": "cartesian",
        "diag_every_n": None,
        "idx": None,
    }

    # Extract starting point
    if "from" in parts:
        from_idx = parts.index("from")
        if from_idx < len(parts) - 1:
            # Get everything between 'from' and 'hessian'
            hess_idx = len(parts)
            if "hessian" in parts:
                hess_idx = parts.index("hessian")
            start_parts = parts[from_idx + 1 : hess_idx]
            metadata["start_point"] = "_".join(start_parts)

    # Extract hessian method
    if "hessian" in parts:
        hess_idx = parts.index("hessian")
        if hess_idx < len(parts) - 1:
            hess_method = parts[hess_idx + 1]
            # Handle case where there might be more parts after hessian
            if hess_method not in ["internal"]:
                metadata["hessian"] = hess_method

    # Extract diag_every_n
    if "diag_every_n" in parts:
        diag_idx = parts.index("diag_every_n")
        if diag_idx < len(parts) - 1:
            diag_every_n = parts[diag_idx + 1]
            metadata["diag_every_n"] = diag_every_n

    # Extract idx
    if "idx" in parts:
        idx_pos = parts.index("idx")
        if idx_pos < len(parts) - 1:
            idx_value = parts[idx_pos + 1]
            metadata["idx"] = idx_value

    # Check for internal coordinates
    if "internal" in parts:
        metadata["coordinates"] = "internal"

    return metadata


def load_all_results(plots_dir: str = "playground/plots_sella") -> pd.DataFrame:
    """Load all Sella TS results from the plots directory."""

    # Find all sella_ts_* directories
    pattern = os.path.join(plots_dir, "sella_ts_*")
    result_dirs = glob.glob(pattern)

    if not result_dirs:
        raise ValueError(f"No sella_ts_* directories found in {plots_dir}")

    print(f"Found {len(result_dirs)} result directories:")
    for d in result_dirs:
        print(f"  - {os.path.basename(d)}")

    all_results = []

    for result_dir in result_dirs:
        summary_file = os.path.join(result_dir, "summary.json")

        if not os.path.exists(summary_file):
            print(f"Warning: No summary.json found in {result_dir}")
            continue

        try:
            with open(summary_file, "r") as f:
                data = json.load(f)

            # Parse experiment metadata from directory name
            dirname = os.path.basename(result_dir)
            metadata = parse_experiment_name(dirname)

            # Combine data with metadata
            result = {"experiment_name": dirname, **metadata, **data}

            all_results.append(result)

        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
            continue

    if not all_results:
        raise ValueError("No valid results found")

    df = pd.DataFrame(all_results)
    print(f"\nLoaded {len(df)} experiments successfully")
    return df


def create_rmsd_plots(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella", suffix: str = ""
):
    """Create RMSD comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"RMSD Analysis Across Sella TS Experiments{suffix}",
        fontsize=10,
        fontweight="bold",
    )

    # Plot 1: RMSD Initial vs Final
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        df["rmsd_initial"],
        df["rmsd_final"],
        c=range(len(df)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    ax1.plot(
        [0, df[["rmsd_initial", "rmsd_final"]].max().max()],
        [0, df[["rmsd_initial", "rmsd_final"]].max().max()],
        "k--",
        alpha=0.5,
        label="y=x",
    )
    ax1.set_xlabel("Initial RMSD (Å)")
    ax1.set_ylabel("Final RMSD (Å)")
    ax1.set_title("Initial vs Final RMSD")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: RMSD Improvement by Starting Point
    ax2 = axes[0, 1]
    start_points = df["start_point"].unique()
    x_pos = np.arange(len(start_points))
    improvements = [
        df[df["start_point"] == sp]["rmsd_improvement"].iloc[0] for sp in start_points
    ]

    bars = ax2.bar(x_pos, improvements, alpha=0.7)
    ax2.set_xlabel("Starting Point")
    ax2.set_ylabel("RMSD Improvement (Å)")
    ax2.set_title("RMSD Improvement by Starting Point")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [sp.replace("_", " ").title() for sp in start_points], rotation=45
    )
    ax2.grid(True, alpha=0.3)

    # Color bars based on improvement (green=good, red=bad)
    for bar, imp in zip(bars, improvements):
        if imp > 0:
            bar.set_color("green")
            bar.set_alpha(0.7)
        else:
            bar.set_color("red")
            bar.set_alpha(0.7)

    # Plot 3: RMSD metrics comparison
    ax3 = axes[1, 0]
    metrics = ["rmsd_initial", "rmsd_final"]
    x = np.arange(len(df))
    width = 0.35

    ax3.bar(x - width / 2, df["rmsd_initial"], width, label="Initial RMSD", alpha=0.7)
    ax3.bar(x + width / 2, df["rmsd_final"], width, label="Final RMSD", alpha=0.7)

    ax3.set_xlabel("Experiment")
    ax3.set_ylabel("RMSD (Å)")
    ax3.set_title("RMSD Initial vs Final by Experiment")
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [
            name.replace("sella_ts_from_", "").replace("_hessian_none", "")
            for name in df["experiment_name"]
        ],
        rotation=45,
        ha="right",
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: RMSD Improvement vs Time Taken
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        df["time_taken"],
        df["rmsd_improvement"],
        c=range(len(df)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    ax4.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="No improvement")
    ax4.set_xlabel("Time Taken (s)")
    ax4.set_ylabel("RMSD Improvement (Å)")
    ax4.set_title("RMSD Improvement vs Computation Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add experiment labels to points
    for i, row in df.iterrows():
        short_name = (
            row["experiment_name"]
            .replace("sella_ts_from_", "")
            .replace("_hessian_none", "")
        )
        ax4.annotate(
            short_name,
            (row["time_taken"], row["rmsd_improvement"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5,
            alpha=0.7,
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, f"rmsd_analysis{suffix}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"RMSD analysis plot saved to: {output_file}")
    # plt.show()


def create_rmsd_standalone_plots(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella", suffix: str = ""
):
    """Create standalone RMSD initial vs final plots - normal and log scale."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"RMSD Initial vs Final - Linear and Log Scale Comparison{suffix}",
        fontsize=10,
        fontweight="bold",
    )

    # Plot 1: RMSD Initial vs Final (Linear Scale)
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        df["rmsd_initial"],
        df["rmsd_final"],
        c=range(len(df)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    ax1.plot(
        [0, df[["rmsd_initial", "rmsd_final"]].max().max()],
        [0, df[["rmsd_initial", "rmsd_final"]].max().max()],
        "k--",
        alpha=0.5,
        label="y=x",
    )
    ax1.set_xlabel("Initial RMSD (Å)")
    ax1.set_ylabel("Final RMSD (Å)")
    ax1.set_title("Initial vs Final RMSD (Linear Scale)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add experiment labels
    for i, row in df.iterrows():
        short_name = (
            row["experiment_name"]
            .replace("sella_ts_from_", "")
            .replace("_hessian_none", "")
        )
        ax1.annotate(
            short_name,
            (row["rmsd_initial"], row["rmsd_final"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5,
            alpha=0.7,
        )

    # Plot 2: RMSD Initial vs Final (Log Scale)
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        df["rmsd_initial"],
        df["rmsd_final"],
        c=range(len(df)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )

    # For log scale, we need to be careful with the y=x line
    # Only plot where both values are positive
    max_val = df[["rmsd_initial", "rmsd_final"]].max().max()
    min_val = max(df[["rmsd_initial", "rmsd_final"]].min().min(), 1e-6)  # Avoid log(0)

    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="y=x",
    )

    ax2.set_xlabel("Initial RMSD (Å)")
    ax2.set_ylabel("Final RMSD (Å)")
    ax2.set_title("Initial vs Final RMSD (Log Scale)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add experiment labels
    for i, row in df.iterrows():
        short_name = (
            row["experiment_name"]
            .replace("sella_ts_from_", "")
            .replace("_hessian_none", "")
        )
        ax2.annotate(
            short_name,
            (row["rmsd_initial"], row["rmsd_final"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5,
            alpha=0.7,
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, f"rmsd_only_comparison{suffix}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Standalone RMSD comparison plot saved to: {output_file}")
    # plt.show()


def create_timing_plots(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella", suffix: str = ""
):
    """Create timing comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Timing Analysis Across Sella TS Experiments{suffix}",
        fontsize=10,
        fontweight="bold",
    )

    # Plot 1: Total optimization time by experiment
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    bars = ax1.bar(x, df["time_taken"], alpha=0.7, color="skyblue")
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("Time Taken (s)")
    ax1.set_title("Total Optimization Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [
            name.replace("sella_ts_from_", "").replace("_hessian_none", "")
            for name in df["experiment_name"]
        ],
        rotation=45,
        ha="right",
    )
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, df["time_taken"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=5,
        )

    # Plot 2: Hessian computation times comparison
    ax2 = axes[0, 1]
    hessian_times = [
        "time_taken_autodiff",
        "time_taken_predict",
        "time_taken_finite_diff",
    ]
    hessian_labels = ["Autodiff", "Predict", "Finite Diff"]

    x = np.arange(len(df))
    width = 0.25

    for i, (time_col, label) in enumerate(zip(hessian_times, hessian_labels)):
        if time_col in df.columns:
            ax2.bar(x + i * width, df[time_col], width, label=label, alpha=0.7)

    ax2.set_xlabel("Experiment")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Hessian Computation Times")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(
        [
            name.replace("sella_ts_from_", "").replace("_hessian_none", "")
            for name in df["experiment_name"]
        ],
        rotation=45,
        ha="right",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")  # Log scale for better visibility

    # Plot 3: Time vs Number of Steps
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        df["nsteps"],
        df["time_taken"],
        c=range(len(df)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    ax3.set_xlabel("Number of Steps")
    ax3.set_ylabel("Time Taken (s)")
    ax3.set_title("Optimization Time vs Number of Steps")
    ax3.grid(True, alpha=0.3)

    # Add experiment labels
    for i, row in df.iterrows():
        short_name = (
            row["experiment_name"]
            .replace("sella_ts_from_", "")
            .replace("_hessian_none", "")
        )
        ax3.annotate(
            short_name,
            (row["nsteps"], row["time_taken"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5,
            alpha=0.7,
        )

    # Plot 4: Time per step analysis
    ax4 = axes[1, 1]
    df["time_per_step"] = df["time_taken"] / df["nsteps"]

    bars = ax4.bar(x, df["time_per_step"], alpha=0.7, color="lightcoral")
    ax4.set_xlabel("Experiment")
    ax4.set_ylabel("Time per Step (s)")
    ax4.set_title("Average Time per Optimization Step")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [
            name.replace("sella_ts_from_", "").replace("_hessian_none", "")
            for name in df["experiment_name"]
        ],
        rotation=45,
        ha="right",
    )
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, time_val in zip(bars, df["time_per_step"]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{time_val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=5,
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, f"timing_analysis{suffix}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Timing analysis plot saved to: {output_file}")
    # plt.show()


def create_steps_plots(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella", suffix: str = ""
):
    """Create number of steps analysis plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Optimization Steps Analysis Across Sella TS Experiments{suffix}",
        fontsize=10,
        fontweight="bold",
    )

    # Plot 1: Number of steps by experiment
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    bars = ax1.bar(x, df["nsteps"], alpha=0.7, color="orange")
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("Number of Steps")
    ax1.set_title("Optimization Steps by Experiment")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [
            name.replace("sella_ts_from_", "").replace("_hessian_none", "")
            for name in df["experiment_name"]
        ],
        rotation=45,
        ha="right",
    )
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, steps in zip(bars, df["nsteps"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{int(steps)}",
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
        )

    # Plot 2: Steps vs RMSD Improvement
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df["nsteps"],
        df["rmsd_improvement"],
        c=range(len(df)),
        cmap="viridis",
        s=150,
        alpha=0.7,
    )
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="No improvement")
    ax2.set_xlabel("Number of Steps")
    ax2.set_ylabel("RMSD Improvement (Å)")
    ax2.set_title("RMSD Improvement vs Number of Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add experiment labels
    for i, row in df.iterrows():
        short_name = (
            row["experiment_name"]
            .replace("sella_ts_from_", "")
            .replace("_hessian_none", "")
        )
        ax2.annotate(
            short_name,
            (row["nsteps"], row["rmsd_improvement"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=6,
            alpha=0.8,
        )

    # Plot 3: Steps by starting point and coordinates
    ax3 = axes[1, 0]

    # Group by start_point and coordinates
    grouped = df.groupby(["start_point", "coordinates"])["nsteps"].first().reset_index()

    # Create combined labels
    grouped["combined_label"] = (
        grouped["start_point"] + "\n(" + grouped["coordinates"] + ")"
    )

    x_pos = np.arange(len(grouped))
    bars = ax3.bar(x_pos, grouped["nsteps"], alpha=0.7)
    ax3.set_xlabel("Starting Point (Coordinate System)")
    ax3.set_ylabel("Number of Steps")
    ax3.set_title("Steps by Starting Point and Coordinate System")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(grouped["combined_label"], rotation=45, ha="right")
    ax3.grid(True, alpha=0.3)

    # Color bars based on coordinate system
    colors = {"cartesian": "lightblue", "internal": "lightcoral"}
    for bar, coord_sys in zip(bars, grouped["coordinates"]):
        bar.set_color(colors.get(coord_sys, "lightgray"))

    # Add value labels
    for bar, steps in zip(bars, grouped["nsteps"]):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 25,
            f"{int(steps)}",
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
        )

    # Plot 4: Steps efficiency (RMSD improvement per step)
    ax4 = axes[1, 1]
    df["rmsd_improvement_per_step"] = df["rmsd_improvement"] / df["nsteps"]

    bars = ax4.bar(
        x, df["rmsd_improvement_per_step"], alpha=0.7, color="mediumseagreen"
    )
    ax4.set_xlabel("Experiment")
    ax4.set_ylabel("RMSD Improvement per Step (Å/step)")
    ax4.set_title("Optimization Efficiency (RMSD Improvement per Step)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [
            name.replace("sella_ts_from_", "").replace("_hessian_none", "")
            for name in df["experiment_name"]
        ],
        rotation=45,
        ha="right",
    )
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # Color bars based on efficiency (green=good, red=bad)
    for bar, eff in zip(bars, df["rmsd_improvement_per_step"]):
        if eff > 0:
            bar.set_color("green")
            bar.set_alpha(0.7)
        else:
            bar.set_color("red")
            bar.set_alpha(0.7)

    # Add value labels
    for bar, eff in zip(bars, df["rmsd_improvement_per_step"]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.00001 if eff > 0 else -0.00002),
            f"{eff:.6f}",
            ha="center",
            va="bottom" if eff > 0 else "top",
            fontsize=5,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, f"steps_analysis{suffix}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Steps analysis plot saved to: {output_file}")
    # plt.show()


def create_summary_table(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella", suffix: str = ""
):
    """Create a summary table of all results."""

    # Select key columns for summary
    summary_cols = [
        "experiment_name",
        "start_point",
        "coordinates",
        "rmsd_initial",
        "rmsd_final",
        "rmsd_improvement",
        "time_taken",
        "nsteps",
        "time_per_step",
    ]

    # Calculate time per step if not already done
    if "time_per_step" not in df.columns:
        df["time_per_step"] = df["time_taken"] / df["nsteps"]

    summary_df = df[summary_cols].copy()

    # Round numerical columns for better readability
    numerical_cols = [
        "rmsd_initial",
        "rmsd_final",
        "rmsd_improvement",
        "time_taken",
        "time_per_step",
    ]
    for col in numerical_cols:
        summary_df[col] = summary_df[col].round(4)

    # Sort by RMSD improvement (best first)
    summary_df = summary_df.sort_values("rmsd_improvement", ascending=False)

    print(f"\n" + "=" * 100)
    print(f"SUMMARY TABLE - Sella TS Experiments{suffix}")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # Save to CSV
    output_file = os.path.join(output_dir, f"sella_results_summary{suffix}.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")

    return summary_df


def create_diag_comparison_plots(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella"
):
    """Create comparison plots across different diag_every_n values."""

    # Filter out rows where diag_every_n is None
    df_with_diag = df[df["diag_every_n"].notna()].copy()

    if len(df_with_diag) == 0:
        print("No experiments with diag_every_n values found for comparison.")
        return

    # Convert diag_every_n to numeric for proper sorting
    df_with_diag["diag_every_n"] = pd.to_numeric(df_with_diag["diag_every_n"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Performance Comparison Across diag_every_n Values",
        fontsize=10,
        fontweight="bold",
    )

    # Plot 1: RMSD Improvement vs diag_every_n
    ax1 = axes[0, 0]
    diag_values = sorted(df_with_diag["diag_every_n"].unique())
    rmsd_improvements = []
    rmsd_stds = []

    for diag_val in diag_values:
        subset = df_with_diag[df_with_diag["diag_every_n"] == diag_val]
        rmsd_improvements.append(subset["rmsd_improvement"].mean())
        rmsd_stds.append(subset["rmsd_improvement"].std() if len(subset) > 1 else 0)

    bars = ax1.bar(
        range(len(diag_values)),
        rmsd_improvements,
        yerr=rmsd_stds,
        capsize=5,
        alpha=0.7,
        color="skyblue",
    )
    ax1.set_xlabel("diag_every_n")
    ax1.set_ylabel("RMSD Improvement (Å)")
    ax1.set_title("Average RMSD Improvement vs diag_every_n")
    ax1.set_xticks(range(len(diag_values)))
    ax1.set_xticklabels([str(int(x)) for x in diag_values])
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # Add value labels on bars
    for bar, improvement in zip(bars, rmsd_improvements):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{improvement:.4f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Plot 2: Time Taken vs diag_every_n
    ax2 = axes[0, 1]
    time_means = []
    time_stds = []

    for diag_val in diag_values:
        subset = df_with_diag[df_with_diag["diag_every_n"] == diag_val]
        time_means.append(subset["time_taken"].mean())
        time_stds.append(subset["time_taken"].std() if len(subset) > 1 else 0)

    bars = ax2.bar(
        range(len(diag_values)),
        time_means,
        yerr=time_stds,
        capsize=5,
        alpha=0.7,
        color="lightcoral",
    )
    ax2.set_xlabel("diag_every_n")
    ax2.set_ylabel("Time Taken (s)")
    ax2.set_title("Average Time Taken vs diag_every_n")
    ax2.set_xticks(range(len(diag_values)))
    ax2.set_xticklabels([str(int(x)) for x in diag_values])
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, time_means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Plot 3: Number of Steps vs diag_every_n
    ax3 = axes[1, 0]
    steps_means = []
    steps_stds = []

    for diag_val in diag_values:
        subset = df_with_diag[df_with_diag["diag_every_n"] == diag_val]
        steps_means.append(subset["nsteps"].mean())
        steps_stds.append(subset["nsteps"].std() if len(subset) > 1 else 0)

    bars = ax3.bar(
        range(len(diag_values)),
        steps_means,
        yerr=steps_stds,
        capsize=5,
        alpha=0.7,
        color="orange",
    )
    ax3.set_xlabel("diag_every_n")
    ax3.set_ylabel("Number of Steps")
    ax3.set_title("Average Number of Steps vs diag_every_n")
    ax3.set_xticks(range(len(diag_values)))
    ax3.set_xticklabels([str(int(x)) for x in diag_values])
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, steps in zip(bars, steps_means):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{int(steps)}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Plot 4: Efficiency (RMSD improvement per step) vs diag_every_n
    ax4 = axes[1, 1]
    efficiency_means = []
    efficiency_stds = []

    for diag_val in diag_values:
        subset = df_with_diag[df_with_diag["diag_every_n"] == diag_val]
        efficiency = subset["rmsd_improvement"] / subset["nsteps"]
        efficiency_means.append(efficiency.mean())
        efficiency_stds.append(efficiency.std() if len(subset) > 1 else 0)

    bars = ax4.bar(
        range(len(diag_values)),
        efficiency_means,
        yerr=efficiency_stds,
        capsize=5,
        alpha=0.7,
        color="mediumseagreen",
    )
    ax4.set_xlabel("diag_every_n")
    ax4.set_ylabel("RMSD Improvement per Step (Å/step)")
    ax4.set_title("Average Efficiency vs diag_every_n")
    ax4.set_xticks(range(len(diag_values)))
    ax4.set_xticklabels([str(int(x)) for x in diag_values])
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # Color bars based on efficiency
    for bar, eff in zip(bars, efficiency_means):
        if eff > 0:
            bar.set_color("green")
            bar.set_alpha(0.7)
        else:
            bar.set_color("red")
            bar.set_alpha(0.7)

    # Add value labels on bars
    for bar, eff in zip(bars, efficiency_means):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.00001 if eff > 0 else -0.00002),
            f"{eff:.6f}",
            ha="center",
            va="bottom" if eff > 0 else "top",
            fontsize=6,
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "diag_every_n_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"diag_every_n comparison plot saved to: {output_file}")

    # Create summary statistics table
    summary_stats = []
    for diag_val in diag_values:
        subset = df_with_diag[df_with_diag["diag_every_n"] == diag_val]
        stats = {
            "diag_every_n": int(diag_val),
            "n_experiments": len(subset),
            "avg_rmsd_improvement": subset["rmsd_improvement"].mean(),
            "std_rmsd_improvement": subset["rmsd_improvement"].std(),
            "avg_time_taken": subset["time_taken"].mean(),
            "std_time_taken": subset["time_taken"].std(),
            "avg_nsteps": subset["nsteps"].mean(),
            "std_nsteps": subset["nsteps"].std(),
            "avg_efficiency": (subset["rmsd_improvement"] / subset["nsteps"]).mean(),
        }
        summary_stats.append(stats)

    stats_df = pd.DataFrame(summary_stats)

    # Save summary statistics
    stats_file = os.path.join(output_dir, "diag_every_n_summary_stats.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"diag_every_n summary statistics saved to: {stats_file}")

    # Print summary
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS BY diag_every_n")
    print("=" * 80)
    print(stats_df.round(6).to_string(index=False))
    print("=" * 80)


def create_idx_comparison_plots(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella"
):
    """Create comparison plots across different idx values."""

    # Filter out rows where idx is None
    df_with_idx = df[df["idx"].notna()].copy()

    if len(df_with_idx) == 0:
        print("No experiments with idx values found for comparison.")
        return

    # Convert idx to numeric for proper sorting if they are numeric
    try:
        df_with_idx["idx"] = pd.to_numeric(df_with_idx["idx"])
        numeric_idx = True
    except (ValueError, TypeError):
        # Keep as string if not numeric
        numeric_idx = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Performance Comparison Across idx Values", fontsize=10, fontweight="bold"
    )

    # Plot 1: RMSD Improvement vs idx
    ax1 = axes[0, 0]
    idx_values = sorted(df_with_idx["idx"].unique())
    rmsd_improvements = []
    rmsd_stds = []

    for idx_val in idx_values:
        subset = df_with_idx[df_with_idx["idx"] == idx_val]
        rmsd_improvements.append(subset["rmsd_improvement"].mean())
        rmsd_stds.append(subset["rmsd_improvement"].std() if len(subset) > 1 else 0)

    bars = ax1.bar(
        range(len(idx_values)),
        rmsd_improvements,
        yerr=rmsd_stds,
        capsize=5,
        alpha=0.7,
        color="skyblue",
    )
    ax1.set_xlabel("idx")
    ax1.set_ylabel("RMSD Improvement (Å)")
    ax1.set_title("Average RMSD Improvement vs idx")
    ax1.set_xticks(range(len(idx_values)))
    if numeric_idx:
        ax1.set_xticklabels([str(int(x)) for x in idx_values])
    else:
        ax1.set_xticklabels([str(x) for x in idx_values])
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # Add value labels on bars
    for bar, improvement in zip(bars, rmsd_improvements):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{improvement:.4f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Plot 2: Time Taken vs idx
    ax2 = axes[0, 1]
    time_means = []
    time_stds = []

    for idx_val in idx_values:
        subset = df_with_idx[df_with_idx["idx"] == idx_val]
        time_means.append(subset["time_taken"].mean())
        time_stds.append(subset["time_taken"].std() if len(subset) > 1 else 0)

    bars = ax2.bar(
        range(len(idx_values)),
        time_means,
        yerr=time_stds,
        capsize=5,
        alpha=0.7,
        color="lightcoral",
    )
    ax2.set_xlabel("idx")
    ax2.set_ylabel("Time Taken (s)")
    ax2.set_title("Average Time Taken vs idx")
    ax2.set_xticks(range(len(idx_values)))
    if numeric_idx:
        ax2.set_xticklabels([str(int(x)) for x in idx_values])
    else:
        ax2.set_xticklabels([str(x) for x in idx_values])
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, time_means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Plot 3: Number of Steps vs idx
    ax3 = axes[1, 0]
    steps_means = []
    steps_stds = []

    for idx_val in idx_values:
        subset = df_with_idx[df_with_idx["idx"] == idx_val]
        steps_means.append(subset["nsteps"].mean())
        steps_stds.append(subset["nsteps"].std() if len(subset) > 1 else 0)

    bars = ax3.bar(
        range(len(idx_values)),
        steps_means,
        yerr=steps_stds,
        capsize=5,
        alpha=0.7,
        color="orange",
    )
    ax3.set_xlabel("idx")
    ax3.set_ylabel("Number of Steps")
    ax3.set_title("Average Number of Steps vs idx")
    ax3.set_xticks(range(len(idx_values)))
    if numeric_idx:
        ax3.set_xticklabels([str(int(x)) for x in idx_values])
    else:
        ax3.set_xticklabels([str(x) for x in idx_values])
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, steps in zip(bars, steps_means):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{int(steps)}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Plot 4: Efficiency (RMSD improvement per step) vs idx
    ax4 = axes[1, 1]
    efficiency_means = []
    efficiency_stds = []

    for idx_val in idx_values:
        subset = df_with_idx[df_with_idx["idx"] == idx_val]
        efficiency = subset["rmsd_improvement"] / subset["nsteps"]
        efficiency_means.append(efficiency.mean())
        efficiency_stds.append(efficiency.std() if len(subset) > 1 else 0)

    bars = ax4.bar(
        range(len(idx_values)),
        efficiency_means,
        yerr=efficiency_stds,
        capsize=5,
        alpha=0.7,
        color="mediumseagreen",
    )
    ax4.set_xlabel("idx")
    ax4.set_ylabel("RMSD Improvement per Step (Å/step)")
    ax4.set_title("Average Efficiency vs idx")
    ax4.set_xticks(range(len(idx_values)))
    if numeric_idx:
        ax4.set_xticklabels([str(int(x)) for x in idx_values])
    else:
        ax4.set_xticklabels([str(x) for x in idx_values])
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # Color bars based on efficiency
    for bar, eff in zip(bars, efficiency_means):
        if eff > 0:
            bar.set_color("green")
            bar.set_alpha(0.7)
        else:
            bar.set_color("red")
            bar.set_alpha(0.7)

    # Add value labels on bars
    for bar, eff in zip(bars, efficiency_means):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.00001 if eff > 0 else -0.00002),
            f"{eff:.6f}",
            ha="center",
            va="bottom" if eff > 0 else "top",
            fontsize=6,
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "idx_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"idx comparison plot saved to: {output_file}")

    # Create summary statistics table
    summary_stats = []
    for idx_val in idx_values:
        subset = df_with_idx[df_with_idx["idx"] == idx_val]
        stats = {
            "idx": str(
                idx_val
            ),  # Keep as string to handle both numeric and non-numeric
            "n_experiments": len(subset),
            "avg_rmsd_improvement": subset["rmsd_improvement"].mean(),
            "std_rmsd_improvement": subset["rmsd_improvement"].std(),
            "avg_time_taken": subset["time_taken"].mean(),
            "std_time_taken": subset["time_taken"].std(),
            "avg_nsteps": subset["nsteps"].mean(),
            "std_nsteps": subset["nsteps"].std(),
            "avg_efficiency": (subset["rmsd_improvement"] / subset["nsteps"]).mean(),
        }
        summary_stats.append(stats)

    stats_df = pd.DataFrame(summary_stats)

    # Save summary statistics
    stats_file = os.path.join(output_dir, "idx_summary_stats.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"idx summary statistics saved to: {stats_file}")

    # Print summary
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS BY idx")
    print("=" * 80)
    print(stats_df.round(6).to_string(index=False))
    print("=" * 80)


def create_transition_state_analysis(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella"
):
    """Create plots analyzing transition state identification across different methods."""

    # Check which frequency analysis columns are available
    freq_methods = []
    for method in ["autodiff", "predict", "finite_diff"]:
        if f"is_transition_state_{method}" in df.columns:
            freq_methods.append(method)

    if not freq_methods:
        print("No transition state analysis data found in results.")
        return

    print(f"Found transition state data for methods: {freq_methods}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Transition State Identification Analysis", fontsize=10, fontweight="bold"
    )

    # Plot 1: Success rate by frequency method
    ax1 = axes[0, 0]
    success_rates = []
    method_names = []

    for method in freq_methods:
        ts_col = f"is_transition_state_{method}"
        if ts_col in df.columns:
            # Count non-null values and successful identifications
            valid_results = df[ts_col].notna()
            if valid_results.sum() > 0:
                success_rate = df.loc[valid_results, ts_col].mean() * 100
                success_rates.append(success_rate)
                method_names.append(method.replace("_", " ").title())

    if success_rates:
        bars = ax1.bar(
            range(len(method_names)),
            success_rates,
            alpha=0.7,
            color=["skyblue", "lightcoral", "lightgreen"][: len(method_names)],
        )
        ax1.set_xlabel("Frequency Analysis Method")
        ax1.set_ylabel("Transition State Success Rate (%)")
        ax1.set_title("TS Identification Success Rate by Method")
        ax1.set_xticks(range(len(method_names)))
        ax1.set_xticklabels(method_names, rotation=45)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
            )

    # Plot 2: Success rate by starting point
    ax2 = axes[0, 1]
    if freq_methods and "start_point" in df.columns:
        start_points = df["start_point"].unique()
        sp_success_rates = []
        sp_labels = []

        for sp in start_points:
            sp_data = df[df["start_point"] == sp]
            # Use the first available method for this analysis
            ts_col = f"is_transition_state_{freq_methods[0]}"
            if ts_col in sp_data.columns:
                valid_results = sp_data[ts_col].notna()
                if valid_results.sum() > 0:
                    success_rate = sp_data.loc[valid_results, ts_col].mean() * 100
                    sp_success_rates.append(success_rate)
                    sp_labels.append(sp.replace("_", " ").title())

        if sp_success_rates:
            bars = ax2.bar(
                range(len(sp_labels)), sp_success_rates, alpha=0.7, color="orange"
            )
            ax2.set_xlabel("Starting Point")
            ax2.set_ylabel("Transition State Success Rate (%)")
            ax2.set_title(f"TS Success Rate by Starting Point ({freq_methods[0]})")
            ax2.set_xticks(range(len(sp_labels)))
            ax2.set_xticklabels(sp_labels, rotation=45, ha="right")
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, rate in zip(bars, sp_success_rates):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    # Plot 3: Success rate by idx (if available)
    ax3 = axes[1, 0]
    if freq_methods and "idx" in df.columns:
        idx_values = sorted([x for x in df["idx"].unique() if x is not None])
        idx_success_rates = []

        for idx_val in idx_values:
            idx_data = df[df["idx"] == idx_val]
            ts_col = f"is_transition_state_{freq_methods[0]}"
            if ts_col in idx_data.columns:
                valid_results = idx_data[ts_col].notna()
                if valid_results.sum() > 0:
                    success_rate = idx_data.loc[valid_results, ts_col].mean() * 100
                    idx_success_rates.append(success_rate)

        if idx_success_rates and len(idx_values) == len(idx_success_rates):
            bars = ax3.bar(
                range(len(idx_values)),
                idx_success_rates,
                alpha=0.7,
                color="mediumseagreen",
            )
            ax3.set_xlabel("idx")
            ax3.set_ylabel("Transition State Success Rate (%)")
            ax3.set_title(f"TS Success Rate by idx ({freq_methods[0]})")
            ax3.set_xticks(range(len(idx_values)))
            ax3.set_xticklabels([str(x) for x in idx_values])
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bar, rate in zip(bars, idx_success_rates):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    # Plot 4: Method agreement analysis
    ax4 = axes[1, 1]
    if len(freq_methods) >= 2:
        # Create agreement matrix
        agreement_data = []

        # Get experiments that have data for multiple methods
        valid_experiments = df.copy()
        for method in freq_methods:
            ts_col = f"is_transition_state_{method}"
            valid_experiments = valid_experiments[valid_experiments[ts_col].notna()]

        if len(valid_experiments) > 0:
            # Calculate agreement statistics
            agreements = []
            labels = []

            if len(freq_methods) >= 2:
                # All methods agree (TS)
                all_ts = True
                for method in freq_methods:
                    ts_col = f"is_transition_state_{method}"
                    all_ts = all_ts & valid_experiments[ts_col]
                agreements.append(all_ts.sum())
                labels.append("All Agree\n(TS)")

                # All methods agree (not TS)
                all_not_ts = True
                for method in freq_methods:
                    ts_col = f"is_transition_state_{method}"
                    all_not_ts = all_not_ts & (~valid_experiments[ts_col])
                agreements.append(all_not_ts.sum())
                labels.append("All Agree\n(Not TS)")

                # Disagreement
                disagreement = len(valid_experiments) - all_ts.sum() - all_not_ts.sum()
                agreements.append(disagreement)
                labels.append("Disagreement")

            # Create pie chart
            colors = ["lightgreen", "lightcoral", "lightyellow"]
            wedges, texts, autotexts = ax4.pie(
                agreements,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            ax4.set_title("Method Agreement Analysis")

            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight("bold")
        else:
            ax4.text(
                0.5,
                0.5,
                "No overlapping\nmethod data",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=5,
            )
            ax4.set_title("Method Agreement Analysis")
    else:
        ax4.text(
            0.5,
            0.5,
            "Need ≥2 methods\nfor agreement analysis",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=5,
        )
        ax4.set_title("Method Agreement Analysis")

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "transition_state_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Transition state analysis plot saved to: {output_file}")

    # Create detailed summary table
    summary_data = []

    for i, row in df.iterrows():
        experiment_summary = {
            "experiment_name": row.get("experiment_name", "unknown"),
            "start_point": row.get("start_point", "unknown"),
            "idx": row.get("idx", "unknown"),
            "rmsd_improvement": row.get("rmsd_improvement", None),
        }

        for method in freq_methods:
            ts_col = f"is_transition_state_{method}"
            neg_freq_col = f"negative_freq_count_{method}"

            if ts_col in row and pd.notna(row[ts_col]):
                experiment_summary[f"is_ts_{method}"] = row[ts_col]
            else:
                experiment_summary[f"is_ts_{method}"] = None

            if neg_freq_col in row and pd.notna(row[neg_freq_col]):
                experiment_summary[f"neg_freq_{method}"] = int(row[neg_freq_col])
            else:
                experiment_summary[f"neg_freq_{method}"] = None

        summary_data.append(experiment_summary)

    summary_df = pd.DataFrame(summary_data)

    # Save detailed summary
    summary_file = os.path.join(output_dir, "transition_state_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Transition state summary saved to: {summary_file}")

    # Print summary statistics
    print(f"\n" + "=" * 80)
    print("TRANSITION STATE IDENTIFICATION SUMMARY")
    print("=" * 80)

    for method in freq_methods:
        ts_col = f"is_ts_{method}"
        if ts_col in summary_df.columns:
            valid_count = summary_df[ts_col].notna().sum()
            ts_count = summary_df[ts_col].sum() if valid_count > 0 else 0
            success_rate = (ts_count / valid_count * 100) if valid_count > 0 else 0
            print(
                f"{method.upper():>12}: {ts_count:>3}/{valid_count:<3} experiments found TS ({success_rate:5.1f}%)"
            )

    print("=" * 80)


def create_hessian_method_analysis(
    df: pd.DataFrame, output_dir: str = "playground/plots_sella"
):
    """Create plots analyzing transition state success rates by hessian method."""

    # Check which frequency analysis columns are available
    freq_methods = []
    for method in ["autodiff", "predict", "finite_diff"]:
        if f"is_transition_state_{method}" in df.columns:
            freq_methods.append(method)

    if not freq_methods:
        print("No transition state analysis data found for hessian method analysis.")
        return

    if "hessian" not in df.columns:
        print("No hessian method information found in the data.")
        return

    print(f"Creating hessian method analysis with TS data from: {freq_methods}")

    # Get unique hessian methods
    hessian_methods = [h for h in df["hessian"].unique() if h is not None]
    if not hessian_methods:
        print("No valid hessian methods found in the data.")
        return

    # Create subplots - one for each frequency method
    n_freq_methods = len(freq_methods)
    if n_freq_methods == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    elif n_freq_methods == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

    fig.suptitle(
        "Transition State Success Rate by Hessian Method",
        fontsize=10,
        fontweight="bold",
    )

    colors = ["skyblue", "lightcoral", "lightgreen", "orange", "mediumpurple"]

    for i, freq_method in enumerate(freq_methods):
        ax = axes[i]
        ts_col = f"is_transition_state_{freq_method}"

        # Calculate success rates for each hessian method
        hessian_success_rates = []
        hessian_labels = []
        hessian_counts = []

        for hess_method in sorted(hessian_methods):
            hess_data = df[df["hessian"] == hess_method]

            if len(hess_data) > 0 and ts_col in hess_data.columns:
                valid_results = hess_data[ts_col].notna()
                if valid_results.sum() > 0:
                    success_rate = hess_data.loc[valid_results, ts_col].mean() * 100
                    hessian_success_rates.append(success_rate)
                    hessian_labels.append(hess_method.replace("_", " ").title())
                    hessian_counts.append(valid_results.sum())

        if hessian_success_rates:
            bars = ax.bar(
                range(len(hessian_labels)),
                hessian_success_rates,
                alpha=0.7,
                color=colors[: len(hessian_labels)],
            )

            ax.set_xlabel("Hessian Method")
            ax.set_ylabel("Transition State Success Rate (%)")
            ax.set_title(
                f"TS Success Rate by Hessian Method\n({freq_method.replace('_', ' ').title()} Frequency Analysis)"
            )
            ax.set_xticks(range(len(hessian_labels)))
            ax.set_xticklabels(hessian_labels, rotation=45, ha="right")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars with counts
            for bar, rate, count in zip(bars, hessian_success_rates, hessian_counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{rate:.1f}%\n(n={count})",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    fontweight="bold",
                )
        else:
            ax.text(
                0.5,
                0.5,
                f"No data available\nfor {freq_method}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title(
                f"TS Success Rate by Hessian Method\n({freq_method.replace('_', ' ').title()} Frequency Analysis)"
            )

    # Hide unused subplots if we have fewer frequency methods than subplots
    if n_freq_methods < len(axes):
        for j in range(n_freq_methods, len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "hessian_method_ts_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Hessian method TS analysis plot saved to: {output_file}")

    # Create summary table
    summary_data = []

    for freq_method in freq_methods:
        ts_col = f"is_transition_state_{freq_method}"

        for hess_method in sorted(hessian_methods):
            hess_data = df[df["hessian"] == hess_method]

            if len(hess_data) > 0 and ts_col in hess_data.columns:
                valid_results = hess_data[ts_col].notna()
                if valid_results.sum() > 0:
                    ts_count = hess_data.loc[valid_results, ts_col].sum()
                    total_count = valid_results.sum()
                    success_rate = (
                        (ts_count / total_count * 100) if total_count > 0 else 0
                    )

                    summary_data.append(
                        {
                            "frequency_method": freq_method,
                            "hessian_method": hess_method,
                            "ts_found": int(ts_count),
                            "total_experiments": int(total_count),
                            "success_rate_percent": round(success_rate, 1),
                        }
                    )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        summary_file = os.path.join(output_dir, "hessian_method_ts_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Hessian method TS summary saved to: {summary_file}")

        # Print summary statistics
        print(f"\n" + "=" * 90)
        print("TRANSITION STATE SUCCESS RATE BY HESSIAN METHOD")
        print("=" * 90)
        print(
            f"{'Frequency Method':<15} {'Hessian Method':<15} {'TS Found':<10} {'Total':<8} {'Success Rate':<12}"
        )
        print("-" * 90)

        for _, row in summary_df.iterrows():
            print(
                f"{row['frequency_method']:<15} {row['hessian_method']:<15} "
                f"{row['ts_found']:<10} {row['total_experiments']:<8} "
                f"{row['success_rate_percent']:<12.1f}%"
            )

        print("=" * 90)

        # Calculate overall best methods
        best_overall = (
            summary_df.groupby("hessian_method")["success_rate_percent"]
            .mean()
            .sort_values(ascending=False)
        )
        print(f"\nOVERALL RANKING BY AVERAGE SUCCESS RATE:")
        for i, (method, rate) in enumerate(best_overall.items(), 1):
            print(f"  {i}. {method}: {rate:.1f}% average success rate")
        print()


def main():
    """Main function to run all analyses."""

    print("Loading Sella TS results...")

    try:
        # Load all results
        df = load_all_results()

        # Create output directory for plots
        output_dir = "playground/plots_sella"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nDataset overview:")
        print(f"  - Number of experiments: {len(df)}")
        print(f"  - Starting points: {df['start_point'].unique()}")
        print(f"  - Coordinate systems: {df['coordinates'].unique()}")
        print(
            f"  - idx values: {sorted([x for x in df['idx'].unique() if x is not None])}"
        )
        print(
            f"  - diag_every_n values: {sorted([x for x in df['diag_every_n'].unique() if x is not None])}"
        )
        print(
            f"  - RMSD improvement range: {df['rmsd_improvement'].min():.4f} to {df['rmsd_improvement'].max():.4f} Å"
        )
        print(
            f"  - Time taken range: {df['time_taken'].min():.1f} to {df['time_taken'].max():.1f} s"
        )

        # Create overall plots
        print("\n" + "=" * 60)
        print("Creating OVERALL plots (all experiments combined)...")
        print("=" * 60)

        print("Creating RMSD analysis plots...")
        create_rmsd_plots(df, output_dir, suffix="_overall")

        print("Creating standalone RMSD comparison plots...")
        create_rmsd_standalone_plots(df, output_dir, suffix="_overall")

        print("Creating timing analysis plots...")
        create_timing_plots(df, output_dir, suffix="_overall")

        print("Creating steps analysis plots...")
        create_steps_plots(df, output_dir, suffix="_overall")

        print("Generating summary table...")
        create_summary_table(df, output_dir, suffix="_overall")

        # Create comparison plots across different diag_every_n values
        print("\nCreating diag_every_n comparison plots...")
        create_diag_comparison_plots(df, output_dir)

        # Create comparison plots across different idx values
        print("\nCreating idx comparison plots...")
        create_idx_comparison_plots(df, output_dir)

        # Create transition state analysis plots
        print("\nCreating transition state analysis plots...")
        create_transition_state_analysis(df, output_dir)

        # Create hessian method analysis plots
        print("\nCreating hessian method transition state analysis plots...")
        create_hessian_method_analysis(df, output_dir)

        # Create plots for each idx value
        idx_values = [x for x in df["idx"].unique() if x is not None]
        idx_values = sorted(idx_values)

        if idx_values:
            print(f"\n" + "=" * 60)
            print(f"Creating SEPARATE plots for each idx value...")
            print(f"Found idx values: {idx_values}")
            print("=" * 60)

            for idx_val in idx_values:
                print(f"\n--- Processing idx = {idx_val} ---")

                # Filter data for this idx value
                df_subset = df[df["idx"] == idx_val].copy()

                if len(df_subset) == 0:
                    print(f"No experiments found for idx = {idx_val}")
                    continue

                print(f"Found {len(df_subset)} experiments for idx = {idx_val}")

                # Create subdirectory
                idx_output_dir = os.path.join(output_dir, f"idx_{idx_val}")
                os.makedirs(idx_output_dir, exist_ok=True)

                print(f"Creating RMSD plots for idx = {idx_val}...")
                create_rmsd_plots(df_subset, idx_output_dir, suffix="")

                print(f"Creating standalone RMSD plots for idx = {idx_val}...")
                create_rmsd_standalone_plots(df_subset, idx_output_dir, suffix="")

                print(f"Creating timing plots for idx = {idx_val}...")
                create_timing_plots(df_subset, idx_output_dir, suffix="")

                print(f"Creating steps plots for idx = {idx_val}...")
                create_steps_plots(df_subset, idx_output_dir, suffix="")

                print(f"Creating summary table for idx = {idx_val}...")
                create_summary_table(df_subset, idx_output_dir, suffix="")

                print(f"Creating transition state analysis for idx = {idx_val}...")
                create_transition_state_analysis(df_subset, idx_output_dir)

                print(f"Creating hessian method analysis for idx = {idx_val}...")
                create_hessian_method_analysis(df_subset, idx_output_dir)

                print(f"✅ Completed plots for idx = {idx_val}")

        # Handle experiments without idx (if any)
        df_no_idx = df[df["idx"].isna()].copy()
        if len(df_no_idx) > 0:
            print(f"\n--- Processing experiments without idx ---")
            print(f"Found {len(df_no_idx)} experiments without idx")

            no_idx_output_dir = os.path.join(output_dir, "no_idx")
            os.makedirs(no_idx_output_dir, exist_ok=True)

            create_rmsd_plots(df_no_idx, no_idx_output_dir, suffix="")
            create_rmsd_standalone_plots(df_no_idx, no_idx_output_dir, suffix="")
            create_timing_plots(df_no_idx, no_idx_output_dir, suffix="")
            create_steps_plots(df_no_idx, no_idx_output_dir, suffix="")
            create_summary_table(df_no_idx, no_idx_output_dir, suffix="")
            create_transition_state_analysis(df_no_idx, no_idx_output_dir)
            create_hessian_method_analysis(df_no_idx, no_idx_output_dir)

            print("✅ Completed plots for experiments without idx")

        print(f"\n🎉 Analysis complete! Check the following directories:")
        print(f"  - Overall plots: {output_dir}")
        print(f"  - Comparison plots: idx_comparison.png, diag_every_n_comparison.png")
        print(f"  - Transition state analysis: transition_state_analysis.png")
        print(f"  - Hessian method analysis: hessian_method_ts_analysis.png")
        for idx_val in idx_values:
            print(f"  - idx = {idx_val}: {os.path.join(output_dir, f'idx_{idx_val}')}")
        if len(df_no_idx) > 0:
            print(f"  - No idx: {os.path.join(output_dir, 'no_idx')}")
        print(f"\nKey files in each directory:")
        print(
            f"  - transition_state_analysis.png: TS identification success rates by method"
        )
        print(f"  - hessian_method_ts_analysis.png: TS success rates by hessian method")
        print(f"  - transition_state_summary.csv: Detailed TS results per experiment")
        print(f"  - hessian_method_ts_summary.csv: TS success rates by hessian method")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
