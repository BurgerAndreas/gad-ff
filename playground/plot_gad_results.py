#!/usr/bin/env python3
"""
Plot GAD (Gentlest Ascent Dynamics) Results

This script loads all GAD transition state search results from the logs_gad directory
and creates comprehensive visualizations comparing RMSD performance across
different eigen methods and test scenarios.

Usage:
    python playground/plot_gad_results.py
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


def parse_test_scenario(scenario_name: str) -> Dict[str, str]:
    """
    Parse test scenario name to extract metadata.

    Example: 'ts_from_r_dt0.01_s1000' ->
    {
        'test_type': 'ts_from_r',
        'dt': '0.01',
        'steps': '1000',
        'description': 'TS from Reactant (dt=0.01, 1000 steps)'
    }
    """
    metadata = {
        "test_type": scenario_name,
        "dt": "unknown",
        "steps": "unknown",
        "description": scenario_name,
    }

    # Extract dt and steps if present
    if "dt" in scenario_name:
        parts = scenario_name.split("_")
        for i, part in enumerate(parts):
            if part.startswith("dt"):
                # Extract dt value (e.g., 'dt0.01' -> '0.01')
                dt_val = part[2:].replace(".", ".")
                metadata["dt"] = dt_val
            elif part.startswith("s") and part[1:].isdigit():
                # Extract steps (e.g., 's1000' -> '1000')
                metadata["steps"] = part[1:]

    # Create human-readable description
    descriptions = {
        "ts_from_ts": "TS from TS (fixed point test)",
        "ts_from_perturbed_ts": "TS from Perturbed TS",
        "ts_from_r_dt0.1_s100": "TS from R (dt=0.1, 100 steps)",
        "ts_from_r_dt0.01_s1000": "TS from R (dt=0.01, 1000 steps)",
        "ts_from_r_dt0.1_s1000": "TS from R (dt=0.1, 1000 steps)",
        "ts_from_r_dt0.1_s10000": "TS from R (dt=0.1, 10k steps)",
        "ts_from_r_p_dt0.01_s1000": "TS from R-P interpolation (dt=0.01, 1000 steps)",
    }

    if scenario_name in descriptions:
        metadata["description"] = descriptions[scenario_name]
        # Extract base test type
        if scenario_name.startswith("ts_from_r_p"):
            metadata["test_type"] = "ts_from_r_p"
        elif scenario_name.startswith("ts_from_r"):
            metadata["test_type"] = "ts_from_r"
        elif scenario_name == "ts_from_ts":
            metadata["test_type"] = "ts_from_ts"
        elif scenario_name == "ts_from_perturbed_ts":
            metadata["test_type"] = "ts_from_perturbed_ts"

    return metadata


def load_all_gad_results(logs_dir: str = "playground/logs_gad") -> pd.DataFrame:
    """Load all GAD results from the logs directory."""

    # Find all results_*.json files
    pattern = os.path.join(logs_dir, "results_*.json")
    result_files = glob.glob(pattern)

    if not result_files:
        raise ValueError(f"No results_*.json files found in {logs_dir}")

    print(f"Found {len(result_files)} GAD result files:")
    for f in result_files:
        print(f"  - {os.path.basename(f)}")

    all_results = []

    for result_file in result_files:
        # Extract eigen method from filename (e.g., results_qr.json -> qr)
        basename = os.path.basename(result_file)
        eigen_method = basename.replace("results_", "").replace(".json", "")

        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            # Convert to long format for easier analysis
            for scenario, rmsd_final in data.items():
                scenario_metadata = parse_test_scenario(scenario)

                result = {
                    "eigen_method": eigen_method,
                    "scenario": scenario,
                    "rmsd_final": rmsd_final,
                    **scenario_metadata,
                }

                all_results.append(result)

        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

    if not all_results:
        raise ValueError("No valid GAD results found")

    df = pd.DataFrame(all_results)
    print(f"\nLoaded {len(df)} GAD experiment results successfully")
    return df


def create_gad_rmsd_plots(df: pd.DataFrame, output_dir: str = "playground/plots_gad"):
    """Create RMSD comparison plots for GAD results."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "GAD RMSD Analysis Across Eigen Methods and Test Scenarios",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: RMSD by Eigen Method (heatmap)
    ax1 = axes[0, 0]

    # Create pivot table for heatmap
    pivot_data = df.pivot(index="scenario", columns="eigen_method", values="rmsd_final")

    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        ax=ax1,
        cbar_kws={"label": "RMSD Final (Å)"},
    )
    ax1.set_title("RMSD Performance Heatmap")
    ax1.set_xlabel("Eigen Method")
    ax1.set_ylabel("Test Scenario")

    # Rotate y-axis labels for better readability
    ax1.set_yticklabels(
        [
            scenario.replace("_", " ").replace("ts from", "TS from")
            for scenario in pivot_data.index
        ],
        rotation=0,
    )

    # Plot 2: RMSD by Eigen Method (grouped bar chart)
    ax2 = axes[0, 1]

    eigen_methods = df["eigen_method"].unique()
    scenarios = df["scenario"].unique()

    x = np.arange(len(eigen_methods))
    width = 0.12  # Width of bars

    colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))

    for i, scenario in enumerate(scenarios):
        scenario_data = df[df["scenario"] == scenario]
        rmsd_values = [
            (
                scenario_data[scenario_data["eigen_method"] == method][
                    "rmsd_final"
                ].iloc[0]
                if len(scenario_data[scenario_data["eigen_method"] == method]) > 0
                else 0
            )
            for method in eigen_methods
        ]

        short_scenario = (
            scenario.replace("ts_from_", "").replace("_dt", " dt").replace("_s", " s")
        )
        ax2.bar(
            x + i * width,
            rmsd_values,
            width,
            label=short_scenario,
            color=colors[i],
            alpha=0.8,
        )

    ax2.set_xlabel("Eigen Method")
    ax2.set_ylabel("RMSD Final (Å)")
    ax2.set_title("RMSD Performance by Eigen Method")
    ax2.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax2.set_xticklabels(eigen_methods, rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best vs Worst Performance
    ax3 = axes[1, 0]

    # Calculate statistics for each eigen method
    method_stats = (
        df.groupby("eigen_method")["rmsd_final"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )

    x_pos = np.arange(len(method_stats))

    # Plot mean with error bars
    ax3.errorbar(
        x_pos,
        method_stats["mean"],
        yerr=method_stats["std"],
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
    )

    # Add min/max markers
    ax3.scatter(
        x_pos,
        method_stats["min"],
        color="green",
        marker="^",
        s=80,
        alpha=0.7,
        label="Best (min)",
    )
    ax3.scatter(
        x_pos,
        method_stats["max"],
        color="red",
        marker="v",
        s=80,
        alpha=0.7,
        label="Worst (max)",
    )

    ax3.set_xlabel("Eigen Method")
    ax3.set_ylabel("RMSD Final (Å)")
    ax3.set_title("Performance Statistics by Eigen Method")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(method_stats["eigen_method"], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Test Scenario Difficulty
    ax4 = axes[1, 1]

    # Calculate average RMSD for each scenario across all methods
    scenario_stats = (
        df.groupby("scenario")["rmsd_final"].agg(["mean", "std"]).reset_index()
    )
    scenario_stats = scenario_stats.sort_values("mean")

    x_pos = np.arange(len(scenario_stats))
    bars = ax4.bar(
        x_pos, scenario_stats["mean"], yerr=scenario_stats["std"], capsize=5, alpha=0.7
    )

    # Color bars based on difficulty (green=easy, red=hard)
    for bar, mean_val in zip(bars, scenario_stats["mean"]):
        if mean_val < 0.5:
            bar.set_color("green")
        elif mean_val < 0.8:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    ax4.set_xlabel("Test Scenario")
    ax4.set_ylabel("Average RMSD Final (Å)")
    ax4.set_title("Test Scenario Difficulty (Lower = Easier)")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(
        [
            s.replace("ts_from_", "").replace("_", " ")
            for s in scenario_stats["scenario"]
        ],
        rotation=45,
        ha="right",
    )
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "gad_rmsd_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"GAD RMSD analysis plot saved to: {output_file}")
    plt.show()


def create_gad_method_comparison(
    df: pd.DataFrame, output_dir: str = "playground/plots_gad"
):
    """Create detailed eigen method comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "GAD Eigen Method Performance Comparison", fontsize=16, fontweight="bold"
    )

    # Plot 1: Method ranking across scenarios
    ax1 = axes[0, 0]

    # For each scenario, rank methods by performance (1 = best)
    ranking_data = []
    for scenario in df["scenario"].unique():
        scenario_df = df[df["scenario"] == scenario].sort_values("rmsd_final")
        for rank, (_, row) in enumerate(scenario_df.iterrows(), 1):
            ranking_data.append(
                {
                    "scenario": scenario,
                    "eigen_method": row["eigen_method"],
                    "rank": rank,
                    "rmsd_final": row["rmsd_final"],
                }
            )

    ranking_df = pd.DataFrame(ranking_data)

    # Create ranking heatmap
    rank_pivot = ranking_df.pivot(
        index="scenario", columns="eigen_method", values="rank"
    )
    sns.heatmap(
        rank_pivot,
        annot=True,
        fmt="d",
        cmap="RdYlGn_r",
        ax=ax1,
        cbar_kws={"label": "Rank (1=Best)"},
    )
    ax1.set_title("Method Ranking by Scenario")
    ax1.set_xlabel("Eigen Method")
    ax1.set_ylabel("Test Scenario")

    # Plot 2: Average ranking
    ax2 = axes[0, 1]

    avg_ranking = ranking_df.groupby("eigen_method")["rank"].mean().sort_values()

    bars = ax2.bar(range(len(avg_ranking)), avg_ranking.values, alpha=0.7)
    ax2.set_xlabel("Eigen Method")
    ax2.set_ylabel("Average Rank (Lower = Better)")
    ax2.set_title("Overall Method Performance Ranking")
    ax2.set_xticks(range(len(avg_ranking)))
    ax2.set_xticklabels(avg_ranking.index, rotation=45)
    ax2.grid(True, alpha=0.3)

    # Color bars based on rank (green=good, red=bad)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels
    for bar, rank in zip(bars, avg_ranking.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{rank:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 3: Consistency analysis (std dev of RMSD)
    ax3 = axes[1, 0]

    method_consistency = df.groupby("eigen_method")["rmsd_final"].std().sort_values()

    bars = ax3.bar(
        range(len(method_consistency)),
        method_consistency.values,
        alpha=0.7,
        color="skyblue",
    )
    ax3.set_xlabel("Eigen Method")
    ax3.set_ylabel("RMSD Standard Deviation (Lower = More Consistent)")
    ax3.set_title("Method Consistency Across Scenarios")
    ax3.set_xticks(range(len(method_consistency)))
    ax3.set_xticklabels(method_consistency.index, rotation=45)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, std_val in zip(bars, method_consistency.values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{std_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 4: Performance vs Complexity (if we had timing data)
    ax4 = axes[1, 1]

    # Show best performance per method for key scenarios
    key_scenarios = ["ts_from_r_dt0.01_s1000", "ts_from_r_p_dt0.01_s1000"]

    methods = df["eigen_method"].unique()
    x_pos = np.arange(len(methods))
    width = 0.35

    for i, scenario in enumerate(key_scenarios):
        scenario_data = df[df["scenario"] == scenario]
        rmsd_values = [
            scenario_data[scenario_data["eigen_method"] == method]["rmsd_final"].iloc[0]
            for method in methods
        ]

        short_name = scenario.replace("ts_from_", "").replace("_dt0.01_s1000", "")
        ax4.bar(x_pos + i * width, rmsd_values, width, label=f"{short_name}", alpha=0.7)

    ax4.set_xlabel("Eigen Method")
    ax4.set_ylabel("RMSD Final (Å)")
    ax4.set_title("Key Scenario Performance Comparison")
    ax4.set_xticks(x_pos + width / 2)
    ax4.set_xticklabels(methods, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "gad_method_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"GAD method comparison plot saved to: {output_file}")
    plt.show()


def create_gad_summary_table(
    df: pd.DataFrame, output_dir: str = "playground/plots_gad"
):
    """Create a summary table of GAD results."""

    # Create pivot table for easy viewing
    summary_pivot = df.pivot(
        index="scenario", columns="eigen_method", values="rmsd_final"
    )

    # Add statistics
    summary_pivot["Mean"] = summary_pivot.mean(axis=1)
    summary_pivot["Std"] = summary_pivot.std(axis=1)
    summary_pivot["Min"] = summary_pivot.min(axis=1)
    summary_pivot["Max"] = summary_pivot.max(axis=1)

    # Round for readability
    summary_pivot = summary_pivot.round(4)

    # Sort by mean performance (best first)
    summary_pivot = summary_pivot.sort_values("Mean")

    print("\n" + "=" * 120)
    print("SUMMARY TABLE - GAD Experiments (RMSD Final Values)")
    print("=" * 120)
    print(summary_pivot.to_string())
    print("=" * 120)

    # Save to CSV
    output_file = os.path.join(output_dir, "gad_results_summary.csv")
    summary_pivot.to_csv(output_file)
    print(f"\nGAD summary table saved to: {output_file}")

    # Create method ranking summary
    method_performance = (
        df.groupby("eigen_method")["rmsd_final"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    method_performance = method_performance.sort_values("mean")

    print("\n" + "=" * 80)
    print("METHOD PERFORMANCE SUMMARY (sorted by average RMSD)")
    print("=" * 80)
    print(method_performance.to_string())
    print("=" * 80)

    # Save method summary
    method_output_file = os.path.join(output_dir, "gad_method_summary.csv")
    method_performance.to_csv(method_output_file)
    print(f"\nMethod summary saved to: {method_output_file}")

    return summary_pivot, method_performance


def main():
    """Main function to run all GAD analyses."""

    print("Loading GAD results...")

    try:
        # Load all results
        df = load_all_gad_results()

        # Create output directory for plots
        output_dir = "playground/plots_gad"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGAD Dataset overview:")
        print(f"  - Number of experiments: {len(df)}")
        print(f"  - Eigen methods: {sorted(df['eigen_method'].unique())}")
        print(f"  - Test scenarios: {len(df['scenario'].unique())}")
        print(
            f"  - RMSD range: {df['rmsd_final'].min():.4f} to {df['rmsd_final'].max():.4f} Å"
        )

        # Identify best and worst performers
        best_result = df.loc[df["rmsd_final"].idxmin()]
        worst_result = df.loc[df["rmsd_final"].idxmax()]
        print(
            f"  - Best result: {best_result['eigen_method']} on {best_result['scenario']} (RMSD: {best_result['rmsd_final']:.4f} Å)"
        )
        print(
            f"  - Worst result: {worst_result['eigen_method']} on {worst_result['scenario']} (RMSD: {worst_result['rmsd_final']:.4f} Å)"
        )

        # Create plots
        print("\nCreating GAD RMSD analysis plots...")
        create_gad_rmsd_plots(df, output_dir)

        print("\nCreating GAD method comparison plots...")
        create_gad_method_comparison(df, output_dir)

        print("\nGenerating GAD summary tables...")
        create_gad_summary_table(df, output_dir)

        print(
            "\n✅ GAD analysis complete! Check the plots_gad directory for visualizations."
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
