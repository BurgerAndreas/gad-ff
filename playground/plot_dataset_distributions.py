#!/usr/bin/env python3
"""
Script to plot data distributions across multiple LMDB datasets.

Creates histograms of the number of atoms and atom type distributions for 1000 random samples from each dataset.
FIXED VERSION: Handles one_hot encoding for atomic numbers and missing datasets.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import os
from collections import Counter
from tqdm import tqdm
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import _fix_dataset_path


# Atomic number to element symbol mapping (based on one_hot encoding indices)
# The one_hot encoding appears to use indices [0, 1, 2, 3, 4] for [H, C, N, O, ?]
ATOMIC_SYMBOLS_FROM_ONEHOT = {
    0: "H",  # Hydrogen
    1: "C",  # Carbon
    2: "N",  # Nitrogen
    3: "O",  # Oxygen
    4: "X",  # Unknown/Other
}


def extract_atomic_numbers_from_onehot(one_hot_tensor):
    """
    Extract atomic number indices from one_hot encoding.

    Args:
        one_hot_tensor: Tensor of shape [n_atoms, n_elements] with one-hot encoding

    Returns:
        List of atomic number indices
    """
    return torch.argmax(one_hot_tensor, dim=1).cpu().numpy().tolist()


def plot_dataset_distributions(
    dataset_files=None, num_samples=1000, output_dir="plots"
):
    """
    Plots histograms of atom counts and atom type distributions for multiple datasets.

    Args:
        dataset_files (list): List of dataset file names to process
        num_samples (int): Number of random samples to analyze per dataset (default: 1000)
        output_dir (str): Output directory for the plots
    """
    if dataset_files is None:
        dataset_files = ["ts1x-val.lmdb", "ts1x-train.lmdb", "RGD1.lmdb", "RGD2.lmdb"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(
        f"Processing {len(dataset_files)} datasets with {num_samples} random samples each"
    )

    # Storage for results
    all_atom_counts = {}
    all_atom_types = {}
    available_datasets = []

    # Process each dataset first to collect data
    for dataset_idx, dataset_file in enumerate(dataset_files):
        print(f"\nProcessing dataset: {dataset_file}")

        try:
            # Load dataset
            input_lmdb_path = _fix_dataset_path(dataset_file)
            dataset = LmdbDataset(input_lmdb_path)
            print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")
            available_datasets.append(dataset_file)

            # Check if dataset has enough samples
            available_samples = len(dataset)
            samples_to_use = min(num_samples, available_samples)

            if available_samples < num_samples:
                print(
                    f"Warning: Dataset has only {available_samples} samples, using all of them"
                )

            # Get random sample indices
            sample_indices = random.sample(range(available_samples), samples_to_use)

            # Extract atom counts and types
            atom_counts = []
            all_atomic_indices = []

            for sample_idx in tqdm(sample_indices, desc=f"Processing {dataset_file}"):
                try:
                    sample = dataset[sample_idx]
                    n_atoms = sample.pos.shape[0]
                    atom_counts.append(n_atoms)

                    # Extract atomic numbers from one_hot encoding
                    if hasattr(sample, "one_hot"):
                        atomic_indices = extract_atomic_numbers_from_onehot(
                            sample.one_hot
                        )
                        all_atomic_indices.extend(atomic_indices)
                    else:
                        print(
                            f"Warning: No one_hot encoding found in sample {sample_idx}"
                        )

                except Exception as e:
                    print(
                        f"Error processing sample {sample_idx} in {dataset_file}: {e}"
                    )
                    continue

            # Store results
            all_atom_counts[dataset_file] = atom_counts
            all_atom_types[dataset_file] = all_atomic_indices

        except Exception as e:
            print(f"Failed to process dataset {dataset_file}: {e}")
            print(f"Skipping {dataset_file}")
            continue

    if not available_datasets:
        print("No datasets could be loaded successfully!")
        return {}, {}

    print(
        f"\nSuccessfully loaded {len(available_datasets)} datasets: {available_datasets}"
    )

    # Create atom count histograms
    print("\nCreating atom count histograms")

    # Dynamic subplot layout based on available datasets
    n_datasets = len(available_datasets)
    if n_datasets == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif n_datasets == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    elif n_datasets <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

    for dataset_idx, dataset_file in enumerate(available_datasets):
        if dataset_file in all_atom_counts:
            atom_counts = all_atom_counts[dataset_file]
            ax = axes[dataset_idx]
            ax.hist(atom_counts, bins=30, alpha=0.7, edgecolor="black")
            ax.set_title(f"{dataset_file}\n({len(atom_counts)} samples)")
            ax.set_xlabel("Number of atoms")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f"Mean: {np.mean(atom_counts):.1f}\nStd: {np.std(atom_counts):.1f}\nMin: {np.min(atom_counts)}\nMax: {np.max(atom_counts)}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    # Hide unused subplots
    for idx in range(len(available_datasets), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    atom_count_file = os.path.join(output_dir, "atom_count_distributions.png")
    plt.savefig(atom_count_file, dpi=300, bbox_inches="tight")
    print(f"Atom count plot saved to: {atom_count_file}")
    plt.close()

    # Create atom type distributions
    print("\nCreating atom type distribution plots")

    # Dynamic subplot layout
    if n_datasets == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif n_datasets == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    elif n_datasets <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

    for dataset_idx, dataset_file in enumerate(available_datasets):
        if dataset_file in all_atom_types and all_atom_types[dataset_file]:
            atomic_indices = all_atom_types[dataset_file]
            ax = axes[dataset_idx]

            # Count occurrences of each atomic index
            atomic_counter = Counter(atomic_indices)

            # Convert to element symbols
            elements = []
            counts = []
            for atomic_idx, count in sorted(atomic_counter.items()):
                element_symbol = ATOMIC_SYMBOLS_FROM_ONEHOT.get(
                    atomic_idx, f"Idx{atomic_idx}"
                )
                elements.append(element_symbol)
                counts.append(count)

            # Create bar plot
            bars = ax.bar(elements, counts, alpha=0.7, edgecolor="black")
            ax.set_title(
                f"{dataset_file}\nElement Distribution ({len(atomic_indices)} total atoms)"
            )
            ax.set_xlabel("Element")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3, axis="y")

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{count}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        else:
            ax = axes[dataset_idx]
            ax.text(
                0.5,
                0.5,
                f"No atom type data\n{dataset_file}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"{dataset_file} (No Data)")

    # Hide unused subplots
    for idx in range(len(available_datasets), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    atom_type_file = os.path.join(output_dir, "atom_type_distributions.png")
    plt.savefig(atom_type_file, dpi=300, bbox_inches="tight")
    print(f"Atom type plot saved to: {atom_type_file}")
    plt.close()

    # Create combined summary plot
    print("\nCreating combined summary plot")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Combined atom count histograms
    for dataset_file in available_datasets:
        if dataset_file in all_atom_counts:
            atom_counts = all_atom_counts[dataset_file]
            ax1.hist(
                atom_counts, bins=20, alpha=0.6, label=dataset_file, edgecolor="black"
            )
    ax1.set_xlabel("Number of atoms")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Atom Count Distribution Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total atoms per dataset
    dataset_names = []
    total_atoms = []
    for dataset_file in available_datasets:
        if dataset_file in all_atom_types and all_atom_types[dataset_file]:
            dataset_names.append(dataset_file.replace(".lmdb", ""))
            total_atoms.append(len(all_atom_types[dataset_file]))

    if dataset_names:
        bars = ax2.bar(dataset_names, total_atoms, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Dataset")
        ax2.set_ylabel("Total Atoms Analyzed")
        ax2.set_title("Total Atoms per Dataset")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        # Add count labels on bars
        for bar, count in zip(bars, total_atoms):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}",
                ha="center",
                va="bottom",
            )

    # Plot 3: Element diversity comparison
    diversity_data = {}
    for dataset_file in available_datasets:
        if dataset_file in all_atom_types and all_atom_types[dataset_file]:
            unique_elements = len(set(all_atom_types[dataset_file]))
            diversity_data[dataset_file.replace(".lmdb", "")] = unique_elements

    if diversity_data:
        names = list(diversity_data.keys())
        diversities = list(diversity_data.values())
        bars = ax3.bar(names, diversities, alpha=0.7, edgecolor="black")
        ax3.set_xlabel("Dataset")
        ax3.set_ylabel("Number of Unique Elements")
        ax3.set_title("Element Diversity per Dataset")
        ax3.grid(True, alpha=0.3, axis="y")
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

        # Add count labels on bars
        for bar, count in zip(bars, diversities):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}",
                ha="center",
                va="bottom",
            )

    # Plot 4: Most common elements across all datasets
    all_elements_combined = []
    for dataset_file in available_datasets:
        if dataset_file in all_atom_types:
            all_elements_combined.extend(all_atom_types[dataset_file])

    if all_elements_combined:
        element_counter = Counter(all_elements_combined)
        top_elements = element_counter.most_common(10)

        elements = [
            ATOMIC_SYMBOLS_FROM_ONEHOT.get(atomic_idx, f"Idx{atomic_idx}")
            for atomic_idx, _ in top_elements
        ]
        counts = [count for _, count in top_elements]

        bars = ax4.bar(elements, counts, alpha=0.7, edgecolor="black")
        ax4.set_xlabel("Element")
        ax4.set_ylabel("Total Count (All Datasets)")
        ax4.set_title("Most Common Elements")
        ax4.grid(True, alpha=0.3, axis="y")

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    summary_file = os.path.join(output_dir, "dataset_summary.png")
    plt.savefig(summary_file, dpi=300, bbox_inches="tight")
    print(f"Summary plot saved to: {summary_file}")
    plt.close()

    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    for dataset_file, atom_counts in all_atom_counts.items():
        if atom_counts:
            print(f"\n{dataset_file}:")
            print(f"  Samples analyzed: {len(atom_counts)}")
            print(f"  Min atoms: {np.min(atom_counts)}")
            print(f"  Max atoms: {np.max(atom_counts)}")
            print(f"  Mean atoms: {np.mean(atom_counts):.2f}")
            print(f"  Std atoms: {np.std(atom_counts):.2f}")
            print(f"  Median atoms: {np.median(atom_counts):.2f}")

            if dataset_file in all_atom_types and all_atom_types[dataset_file]:
                atomic_indices = all_atom_types[dataset_file]
                unique_elements = set(atomic_indices)
                element_symbols = [
                    ATOMIC_SYMBOLS_FROM_ONEHOT.get(idx, f"Idx{idx}")
                    for idx in sorted(unique_elements)
                ]
                print(f"  Total atoms: {len(atomic_indices)}")
                print(f"  Unique elements: {len(unique_elements)}")
                print(f"  Elements present: {', '.join(element_symbols)}")

    return all_atom_counts, all_atom_types


if __name__ == "__main__":
    """
    python playground/plot_dataset_distributions_fixed.py
    python playground/plot_dataset_distributions_fixed.py --dataset-files ts1x-val.lmdb RGD1.lmdb --num-samples 1000
    """
    parser = argparse.ArgumentParser(
        description="Plot data distributions across multiple LMDB datasets (FIXED VERSION)"
    )
    parser.add_argument(
        "--dataset-files",
        type=str,
        nargs="+",
        default=["ts1x-val.lmdb", "ts1x-train.lmdb", "RGD1.lmdb", "RGD2.lmdb"],
        help="List of dataset files to process (default: ts1x-val.lmdb ts1x-train.lmdb RGD1.lmdb RGD2.lmdb)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of random samples to analyze per dataset (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for the plots (default: plots)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    plot_dataset_distributions(
        dataset_files=args.dataset_files,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )
