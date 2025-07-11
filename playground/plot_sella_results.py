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


def parse_experiment_name(dirname: str) -> Dict[str, str]:
    """
    Parse experiment directory name to extract metadata.

    Example: 'sella_ts_from_linear_r_p_hessian_none' ->
    {
        'method': 'sella_ts',
        'start_point': 'linear_r_p',
        'hessian': 'none',
        'coordinates': 'cartesian'
    }
    """
    parts = dirname.split("_")

    metadata = {
        "method": "sella_ts",
        "start_point": "unknown",
        "hessian": "none",
        "coordinates": "cartesian",
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

    # Check for internal coordinates
    if "internal" in parts:
        metadata["coordinates"] = "internal"

    return metadata


def load_all_results(plots_dir: str = "playground/plots") -> pd.DataFrame:
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


def create_rmsd_plots(df: pd.DataFrame, output_dir: str = "playground/plots_sella"):
    """Create RMSD comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "RMSD Analysis Across Sella TS Experiments", fontsize=16, fontweight="bold"
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
            fontsize=8,
            alpha=0.7,
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "rmsd_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"RMSD analysis plot saved to: {output_file}")
    # plt.show()


def create_timing_plots(df: pd.DataFrame, output_dir: str = "playground/plots_sella"):
    """Create timing comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Timing Analysis Across Sella TS Experiments", fontsize=16, fontweight="bold"
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
            fontsize=8,
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
            fontsize=8,
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
            fontsize=8,
        )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "timing_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Timing analysis plot saved to: {output_file}")
    # plt.show()


def create_steps_plots(df: pd.DataFrame, output_dir: str = "playground/plots_sella"):
    """Create number of steps analysis plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimization Steps Analysis Across Sella TS Experiments', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of steps by experiment
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    bars = ax1.bar(x, df['nsteps'], alpha=0.7, color='orange')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Number of Steps')
    ax1.set_title('Optimization Steps by Experiment')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace('sella_ts_from_', '').replace('_hessian_none', '') 
                        for name in df['experiment_name']], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, steps in zip(bars, df['nsteps']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{int(steps)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Steps vs RMSD Improvement
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['nsteps'], df['rmsd_improvement'], 
                         c=range(len(df)), cmap='viridis', s=150, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No improvement')
    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('RMSD Improvement (Å)')
    ax2.set_title('RMSD Improvement vs Number of Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add experiment labels
    for i, row in df.iterrows():
        short_name = row['experiment_name'].replace('sella_ts_from_', '').replace('_hessian_none', '')
        ax2.annotate(short_name, (row['nsteps'], row['rmsd_improvement']), 
                    xytext=(10, 10), textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Plot 3: Steps by starting point and coordinates
    ax3 = axes[1, 0]
    
    # Group by start_point and coordinates
    grouped = df.groupby(['start_point', 'coordinates'])['nsteps'].first().reset_index()
    
    # Create combined labels
    grouped['combined_label'] = grouped['start_point'] + '\n(' + grouped['coordinates'] + ')'
    
    x_pos = np.arange(len(grouped))
    bars = ax3.bar(x_pos, grouped['nsteps'], alpha=0.7)
    ax3.set_xlabel('Starting Point (Coordinate System)')
    ax3.set_ylabel('Number of Steps')
    ax3.set_title('Steps by Starting Point and Coordinate System')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(grouped['combined_label'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Color bars based on coordinate system
    colors = {'cartesian': 'lightblue', 'internal': 'lightcoral'}
    for bar, coord_sys in zip(bars, grouped['coordinates']):
        bar.set_color(colors.get(coord_sys, 'lightgray'))
    
    # Add value labels
    for bar, steps in zip(bars, grouped['nsteps']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25, 
                f'{int(steps)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Steps efficiency (RMSD improvement per step)
    ax4 = axes[1, 1]
    df['rmsd_improvement_per_step'] = df['rmsd_improvement'] / df['nsteps']
    
    bars = ax4.bar(x, df['rmsd_improvement_per_step'], alpha=0.7, color='mediumseagreen')
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('RMSD Improvement per Step (Å/step)')
    ax4.set_title('Optimization Efficiency (RMSD Improvement per Step)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.replace('sella_ts_from_', '').replace('_hessian_none', '') 
                        for name in df['experiment_name']], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Color bars based on efficiency (green=good, red=bad)
    for bar, eff in zip(bars, df['rmsd_improvement_per_step']):
        if eff > 0:
            bar.set_color('green')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    # Add value labels
    for bar, eff in zip(bars, df['rmsd_improvement_per_step']):
        ax4.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.00001 if eff > 0 else -0.00002), 
                f'{eff:.6f}', ha='center', va='bottom' if eff > 0 else 'top', 
                fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, "steps_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Steps analysis plot saved to: {output_file}")
    # plt.show()


def create_summary_table(df: pd.DataFrame, output_dir: str = "playground/plots_sella"):
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

    print("\n" + "=" * 100)
    print("SUMMARY TABLE - Sella TS Experiments")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # Save to CSV
    output_file = os.path.join(output_dir, "sella_results_summary.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")

    return summary_df


def main():
    """Main function to run all analyses."""

    print("Loading Sella TS results...")

    try:
        # Load all results
        df = load_all_results()

        # Create output directory for plots
        output_dir = "playground/plots"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nDataset overview:")
        print(f"  - Number of experiments: {len(df)}")
        print(f"  - Starting points: {df['start_point'].unique()}")
        print(f"  - Coordinate systems: {df['coordinates'].unique()}")
        print(
            f"  - RMSD improvement range: {df['rmsd_improvement'].min():.4f} to {df['rmsd_improvement'].max():.4f} Å"
        )
        print(
            f"  - Time taken range: {df['time_taken'].min():.1f} to {df['time_taken'].max():.1f} s"
        )

        # Create plots
        print("\nCreating RMSD analysis plots...")
        create_rmsd_plots(df, output_dir)

        print("\nCreating timing analysis plots...")
        create_timing_plots(df, output_dir)

        print("\nCreating steps analysis plots...")
        create_steps_plots(df, output_dir)

        print("\nGenerating summary table...")
        create_summary_table(df, output_dir)

        print("\n✅ Analysis complete! Check the plots directory for visualizations.")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
