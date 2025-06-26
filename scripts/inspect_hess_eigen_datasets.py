#!/usr/bin/env python3
"""
Script to inspect and visualize the distribution of Hessian eigenvalues in the HORM eigen datasets.

Creates histograms for eigenvalue 1, eigenvalue 2, and their joint distribution.
Analyzes index-1 saddle point statistics across different datasets.
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch_geometric.loader import DataLoader as TGDataLoader

from gadff.horm.ff_lmdb import LmdbDataset, fix_hessian_eigen_transform
from gadff.path_config import DATASET_DIR_HORM_EIGEN, DATASET_FILES_HORM


def load_eigenvalues_from_dataset(dataset_path, max_samples=None, flip_sign=False):
    """
    Load eigenvalues from an eigen dataset.

    Args:
        dataset_path: Path to the eigen dataset (.lmdb file)
        max_samples: Maximum number of samples to load (None for all)
        flip_sign: Flip the sign of the Hessian
    Returns:
        tuple: (eigenvals_1, eigenvals_2) as numpy arrays
    """
    print(f"Loading dataset: {dataset_path}")

    try:
        dataset = LmdbDataset(dataset_path, transform=fix_hessian_eigen_transform)
        print(f"  Found {len(dataset)} samples")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return None, None

    # Determine sample size
    num_samples = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )
    print(f"  Loading {num_samples} samples...")

    eigenvals_1 = []
    eigenvals_2 = []

    # Load eigenvalues
    for i in tqdm(range(num_samples), desc="Loading eigenvalues"):
        try:
            sample = dataset[i]

            # Extract eigenvalues
            if hasattr(sample, "hessian_eigenvalue_1") and hasattr(
                sample, "hessian_eigenvalue_2"
            ):
                ev1 = sample.hessian_eigenvalue_1.item()
                ev2 = sample.hessian_eigenvalue_2.item()
                if flip_sign:
                    ev1 = -ev1
                    ev2 = -ev2
                eigenvals_1.append(ev1)
                eigenvals_2.append(ev2)
            else:
                print(f"  Warning: Sample {i} missing eigenvalue data")

        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
            continue

    print(f"  Successfully loaded {len(eigenvals_1)} eigenvalue pairs")

    return np.array(eigenvals_1), np.array(eigenvals_2)


def compute_statistics(eigenvals_1, eigenvals_2, dataset_name):
    """
    Compute comprehensive statistics for eigenvalues.

    Args:
        eigenvals_1: Array of first eigenvalues
        eigenvals_2: Array of second eigenvalues
        dataset_name: Name of the dataset for reporting

    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        "dataset": dataset_name,
        "n_samples": len(eigenvals_1),
    }

    # Basic statistics for eigenvalue 1
    stats["ev1_mean"] = np.mean(eigenvals_1)
    stats["ev1_std"] = np.std(eigenvals_1)
    stats["ev1_min"] = np.min(eigenvals_1)
    stats["ev1_max"] = np.max(eigenvals_1)
    stats["ev1_median"] = np.median(eigenvals_1)

    # Basic statistics for eigenvalue 2
    stats["ev2_mean"] = np.mean(eigenvals_2)
    stats["ev2_std"] = np.std(eigenvals_2)
    stats["ev2_min"] = np.min(eigenvals_2)
    stats["ev2_max"] = np.max(eigenvals_2)
    stats["ev2_median"] = np.median(eigenvals_2)

    # Sign statistics
    stats["ev1_positive"] = np.sum(eigenvals_1 > 0)
    stats["ev1_negative"] = np.sum(eigenvals_1 < 0)
    stats["ev1_zero"] = np.sum(eigenvals_1 == 0)

    stats["ev2_positive"] = np.sum(eigenvals_2 > 0)
    stats["ev2_negative"] = np.sum(eigenvals_2 < 0)
    stats["ev2_zero"] = np.sum(eigenvals_2 == 0)

    # Index-1 saddle point analysis (one negative, one positive eigenvalue)
    is_index1_saddle = ((eigenvals_1 < 0) & (eigenvals_2 > 0)) | (
        (eigenvals_1 > 0) & (eigenvals_2 < 0)
    )
    stats["index1_saddle_count"] = np.sum(is_index1_saddle)
    stats["index1_saddle_fraction"] = stats["index1_saddle_count"] / len(eigenvals_1)

    # Both negative (local maximum)
    both_negative = (eigenvals_1 < 0) & (eigenvals_2 < 0)
    stats["both_negative_count"] = np.sum(both_negative)
    stats["both_negative_fraction"] = stats["both_negative_count"] / len(eigenvals_1)

    # Both positive (local minimum)
    both_positive = (eigenvals_1 > 0) & (eigenvals_2 > 0)
    stats["both_positive_count"] = np.sum(both_positive)
    stats["both_positive_fraction"] = stats["both_positive_count"] / len(eigenvals_1)

    return stats


def setup_plotting():
    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")


def plot_individual_histograms(eigenvals_1, eigenvals_2, dataset_name, output_dir):
    """Create individual histograms for eigenvalue 1 and eigenvalue 2."""
    setup_plotting()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Eigenvalue 1 histogram
    ax1.hist(eigenvals_1, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax1.axvline(0, color="red", linestyle="--", alpha=0.8, label="Zero")
    ax1.set_xlabel("Eigenvalue 1")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{dataset_name}: Eigenvalue 1 Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Eigenvalue 2 histogram
    ax2.hist(eigenvals_2, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax2.axvline(0, color="red", linestyle="--", alpha=0.8, label="Zero")
    ax2.set_xlabel("Eigenvalue 2")
    ax2.set_ylabel("Density")
    ax2.set_title(f"{dataset_name}: Eigenvalue 2 Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output_dir}/{dataset_name}_eigenvalue_histograms.png"
    print(f" Saved {fname}")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_joint_distribution_2d(eigenvals_1, eigenvals_2, dataset_name, output_dir):
    """Create 2D histogram for joint distribution of eigenvalues."""
    setup_plotting()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create 2D histogram
    h = ax.hist2d(eigenvals_1, eigenvals_2, bins=150, density=True, cmap="Blues")
    plt.colorbar(h[3], ax=ax, label="Density")

    ax.set_xlim(-50, 5)
    ax.set_ylim(-3, 1)

    ax.set_xlabel("Eigenvalue 1")
    ax.set_ylabel("Eigenvalue 2")
    ax.set_title(f"{dataset_name}: Joint Distribution of Eigenvalues")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output_dir}/{dataset_name}_joint_distribution.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f" Saved {fname}")
    plt.close()


def plot_logx_histograms(eigenvals_1, eigenvals_2, dataset_name, output_dir):
    """Create histograms with log-scale x-axis (eigenvalue magnitudes)."""
    setup_plotting()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Log-scale histogram for eigenvalue 1
    ev1_pos = eigenvals_1[eigenvals_1 > 0]
    ev1_neg = eigenvals_1[eigenvals_1 < 0]

    if len(ev1_pos) > 0:
        ax1.hist(
            np.log10(ev1_pos),
            bins=30,
            alpha=0.7,
            label="Positive (log10)",
            color="blue",
        )
    if len(ev1_neg) > 0:
        ax1.hist(
            np.log10(-ev1_neg),
            bins=30,
            alpha=0.7,
            label="Negative (log10 of |val|)",
            color="red",
        )

    ax1.set_xlabel("log10(|Eigenvalue 1|)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{dataset_name}: Eigenvalue 1 Log Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log-scale histogram for eigenvalue 2
    ev2_pos = eigenvals_2[eigenvals_2 > 0]
    ev2_neg = eigenvals_2[eigenvals_2 < 0]

    if len(ev2_pos) > 0:
        ax2.hist(
            np.log10(ev2_pos),
            bins=30,
            alpha=0.7,
            label="Positive (log10)",
            color="blue",
        )
    if len(ev2_neg) > 0:
        ax2.hist(
            np.log10(-ev2_neg),
            bins=30,
            alpha=0.7,
            label="Negative (log10 of |val|)",
            color="red",
        )

    ax2.set_xlabel("log10(|Eigenvalue 2|)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{dataset_name}: Eigenvalue 2 Log Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output_dir}/{dataset_name}_logx_histograms.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f" Saved {fname}")
    plt.close()


def plot_logy_histograms(eigenvals_1, eigenvals_2, dataset_name, output_dir):
    """Create histograms with log-scale y-axis (counts)."""
    setup_plotting()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Log-scale histogram for eigenvalue 1 (y-axis log scale)
    ax1.hist(eigenvals_1, bins=50, alpha=0.7, edgecolor="black")
    ax1.axvline(0, color="red", linestyle="--", alpha=0.8, label="Zero")
    ax1.set_xlabel("Eigenvalue 1")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_yscale("log")
    ax1.set_title(f"{dataset_name}: Eigenvalue 1 Distribution (Log Y-axis)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log-scale histogram for eigenvalue 2 (y-axis log scale)
    ax2.hist(eigenvals_2, bins=50, alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--", alpha=0.8, label="Zero")
    ax2.set_xlabel("Eigenvalue 2")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_yscale("log")
    ax2.set_title(f"{dataset_name}: Eigenvalue 2 Distribution (Log Y-axis)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output_dir}/{dataset_name}_logy_histograms.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f" Saved {fname}")
    plt.close()


def plot_kde_joint_distribution(eigenvals_1, eigenvals_2, dataset_name, output_dir):
    """Create KDE plot for joint distribution of eigenvalues."""
    setup_plotting()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create KDE plot for joint distribution with error handling
    try:
        # Option 1: Try with fewer levels first
        sns.kdeplot(
            x=eigenvals_1, y=eigenvals_2, ax=ax, fill=True, cmap="Blues", alpha=0.7
        )
        sns.kdeplot(
            x=eigenvals_1,
            y=eigenvals_2,
            ax=ax,
            levels=5,
            colors="black",
            alpha=0.5,
            linewidths=1,
        )
    except Exception as e:
        print(f"  Warning: Could not create KDE plot: {e}")
        return None

    ax.set_xlim(-50, 5)
    ax.set_ylim(-3, 1)

    ax.set_xlabel("Eigenvalue 1")
    ax.set_ylabel("Eigenvalue 2")
    ax.set_title(f"{dataset_name}: KDE Joint Distribution of Eigenvalues")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output_dir}/{dataset_name}_kde_joint.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f" Saved {fname}")
    plt.close()


def plot_scatter_joint_distribution(eigenvals_1, eigenvals_2, dataset_name, output_dir):
    """Create scatter plot for joint distribution of eigenvalues."""
    setup_plotting()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # scatter plot with transparency
    ax.scatter(eigenvals_1, eigenvals_2, alpha=0.1, s=5, c="blue")

    ax.set_xlim(-50, 5)
    ax.set_ylim(-3, 1)

    ax.set_xlabel("Eigenvalue 1")
    ax.set_ylabel("Eigenvalue 2")
    ax.set_title(f"{dataset_name}: Scatter Joint Distribution of Eigenvalues")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output_dir}/{dataset_name}_scatter_joint.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f" Saved {fname}")
    plt.close()


def create_eigenvalue_plots(
    eigenvals_1, eigenvals_2, dataset_name, output_dir="eigenvalue_plots"
):
    """
    Create comprehensive plots for eigenvalue distributions.

    Args:
        eigenvals_1: Array of first eigenvalues
        eigenvals_2: Array of second eigenvalues
        dataset_name: Name of the dataset for plot titles
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = dataset_name.split("/")[-1]

    # Create all plots
    plot_individual_histograms(eigenvals_1, eigenvals_2, dataset_name, output_dir)
    # plot_logx_histograms(eigenvals_1, eigenvals_2, dataset_name, output_dir)
    plot_logy_histograms(eigenvals_1, eigenvals_2, dataset_name, output_dir)
    plot_joint_distribution_2d(eigenvals_1, eigenvals_2, dataset_name, output_dir)
    plot_kde_joint_distribution(eigenvals_1, eigenvals_2, dataset_name, output_dir)
    plot_scatter_joint_distribution(eigenvals_1, eigenvals_2, dataset_name, output_dir)

    print(f"  All plots saved to {output_dir}/")


def print_statistics_table(all_stats):
    """
    Print a formatted table of statistics for all datasets.

    Args:
        all_stats: List of statistics dictionaries from compute_statistics
    """
    print("\n" + "=" * 100)
    print("EIGENVALUE STATISTICS SUMMARY")
    print("=" * 100)

    # Header
    header = f"{'Dataset':<20} {'Samples':<10} {'EV1 Mean':<12} {'EV1 Std':<12} {'EV2 Mean':<12} {'EV2 Std':<12} {'Saddle %':<10}"
    print(header)
    print("-" * len(header))

    # Data rows
    for stats in all_stats:
        row = f"{stats['dataset']:<20} {stats['n_samples']:<10} {stats['ev1_mean']:<12.4f} {stats['ev1_std']:<12.4f} {stats['ev2_mean']:<12.4f} {stats['ev2_std']:<12.4f} {stats['index1_saddle_fraction']*100:<10.2f}"
        print(row)

    print("\n" + "=" * 100)
    print("DETAILED STATISTICS")
    print("=" * 100)

    for stats in all_stats:
        print(f"\n{stats['dataset']} ({stats['n_samples']} samples):")
        print(
            f"  Eigenvalue 1: mean={stats['ev1_mean']:.4f}, std={stats['ev1_std']:.4f}, range=[{stats['ev1_min']:.4f}, {stats['ev1_max']:.4f}]"
        )
        print(
            f"  Eigenvalue 2: mean={stats['ev2_mean']:.4f}, std={stats['ev2_std']:.4f}, range=[{stats['ev2_min']:.4f}, {stats['ev2_max']:.4f}]"
        )
        print(f"  Sign distribution:")
        print(
            f"    EV1: {stats['ev1_positive']} pos, {stats['ev1_negative']} neg, {stats['ev1_zero']} zero"
        )
        print(
            f"    EV2: {stats['ev2_positive']} pos, {stats['ev2_negative']} neg, {stats['ev2_zero']} zero"
        )
        print(f"  Critical point types:")
        print(
            f"    Index-1 saddles: {stats['index1_saddle_count']} ({stats['index1_saddle_fraction']*100:.2f}%)"
        )
        print(
            f"    Local minima (both pos): {stats['both_positive_count']} ({stats['both_positive_fraction']*100:.2f}%)"
        )
        print(
            f"    Local maxima (both neg): {stats['both_negative_count']} ({stats['both_negative_fraction']*100:.2f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect eigenvalue distributions in HORM eigen datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "data/sample_100-dft-hess-eigen.lmdb",
            "ts1x-val-dft-hess-eigen.lmdb",
            "RGD1-dft-hess-eigen.lmdb",
        ],
        help="Dataset files to analyze (default: ts1x-val-dft-hess-eigen.lmdb RGD1-dft-hess-eigen.lmdb)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load per dataset (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="eigenvalue_plots",
        help="Directory to save plots (default: eigenvalue_plots)",
    )
    parser.add_argument(
        "--data-dir",
        default=DATASET_DIR_HORM_EIGEN,
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--flip-sign", action="store_true", help="Flip the sign of the Hessian"
    )
    args = parser.parse_args()

    print("HORM Eigenvalue Dataset Inspector")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples per dataset: {args.max_samples or 'All'}")
    print(f"Datasets to analyze: {args.datasets}")

    all_stats = []

    # Process each dataset
    for dataset_file in args.datasets:
        if os.path.exists(dataset_file):
            dataset_path = os.path.abspath(dataset_file)
        else:
            dataset_path = os.path.join(args.data_dir, dataset_file)
        dataset_name = dataset_file.replace("-equ-hess-eigen.lmdb", "").replace(
            ".lmdb", ""
        )
        dataset_name = dataset_file.replace("-dft-hess-eigen.lmdb", "").replace(
            ".lmdb", ""
        )

        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")

        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset file not found: {dataset_path}")
            continue

        # Load eigenvalues
        eigenvals_1, eigenvals_2 = load_eigenvalues_from_dataset(
            dataset_path, args.max_samples, args.flip_sign
        )

        if eigenvals_1 is None or len(eigenvals_1) == 0:
            print(f"Error: No eigenvalues loaded from {dataset_file}")
            continue

        # Compute statistics
        stats = compute_statistics(eigenvals_1, eigenvals_2, dataset_name)
        all_stats.append(stats)

        # Create plots
        create_eigenvalue_plots(eigenvals_1, eigenvals_2, dataset_name, args.output_dir)

    # Print summary table
    if all_stats:
        print_statistics_table(all_stats)
    else:
        print("No datasets were successfully processed.")

    print(f"\nAnalysis complete. Plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    """Example usage:
    python scripts/inspect_hess_eigen_datasets.py --datasets RGD1-dft-hess-eigen.lmdb
    python scripts/inspect_hess_eigen_datasets.py --datasets RGD1-dft-hess-eigen.lmdb --flip-sign
    python scripts/inspect_hess_eigen_datasets.py --datasets data/sample_100-dft-hess-eigen.lmdb ts1x-val-dft-hess-eigen.lmdb RGD1-dft-hess-eigen.lmdb ts1x_hess_train_big-dft-hess-eigen.lmdb
    python scripts/inspect_hess_eigen_datasets.py --datasets data/sample_100-equ-hess-eigen.lmdb ts1x-val-equ-hess-eigen.lmdb RGD1-equ-hess-eigen.lmdb
    """
    main()
