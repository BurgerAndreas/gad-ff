#!/usr/bin/env python3
"""
Script to inspect and compare eigenvalue computations on DFT Hessians.

Computes eigenvalues of the Hessian for the first 100 samples using both torch.linalg.eig and torch.linalg.eigh.
Checks ordering, compares differences between methods, and analyzes Hessian asymmetry errors.
"""
import argparse
import numpy as np
import time
import torch
from tqdm import tqdm
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import _fix_dataset_path


def inspect_dft_hess_eigen(
    dataset_file="ts1x-val.lmdb", num_samples=1000, flip_sign=False
):
    """
    Inspects eigenvalue computations on DFT Hessians for the first num_samples.

    Args:
        dataset_file (str): Name of the dataset file to process
        num_samples (int): Number of samples to analyze (default: 100)
    """
    # ---- Load dataset ----
    input_lmdb_path = _fix_dataset_path(dataset_file)
    dataset = LmdbDataset(input_lmdb_path)
    print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")

    # Check if hessian field exists
    first_sample = dataset[0]
    if not hasattr(first_sample, "hessian"):
        raise ValueError(
            "Dataset does not contain 'hessian' field. Cannot compute DFT eigenvalues."
        )

    print(f"Processing first {num_samples} samples...")

    # Storage for results
    eig_eigenvals_all = []
    eigh_eigenvals_all = []
    eigenval_diffs = []
    eigenvec_diffs = []
    asymmetry_errors = []
    eig_ordered_correctly = []
    eigh_ordered_correctly = []

    # ---- Main analysis loop ----
    for sample_idx in tqdm(
        range(min(num_samples, len(dataset))), desc="Processing samples"
    ):
        try:
            # Get sample and extract hessian
            sample = dataset[sample_idx]
            dft_hessian = sample.hessian
            n_atoms = sample.pos.shape[0]

            # Reshape hessian to [3*N, 3*N] format
            hessian_matrix = dft_hessian.reshape(n_atoms * 3, n_atoms * 3)

            if flip_sign:
                hessian_matrix = -hessian_matrix

            # Check asymmetry error
            asymmetry_error = torch.max(
                torch.abs(hessian_matrix - hessian_matrix.T)
            ).item()
            asymmetry_errors.append(asymmetry_error)

            # Compute eigenvalues using both methods
            # torch.linalg.eig returns complex eigenvalues, but for symmetric matrices they should be real
            eig_eigenvals, eig_eigenvecs = torch.linalg.eig(hessian_matrix)
            eigh_eigenvals, eigh_eigenvecs = torch.linalg.eigh(hessian_matrix)

            # Convert complex eigenvalues to real (should have zero imaginary part for symmetric matrices)
            eig_eigenvals_real = eig_eigenvals.real

            # Store all eigenvalues for comparison
            eig_eigenvals_all.append(eig_eigenvals_real.cpu().numpy())
            eigh_eigenvals_all.append(eigh_eigenvals.cpu().numpy())

            # Check if eigenvalues are ordered (smallest first)
            eig_sorted = torch.all(eig_eigenvals_real[:-1] <= eig_eigenvals_real[1:])
            eigh_sorted = torch.all(eigh_eigenvals[:-1] <= eigh_eigenvals[1:])
            eig_ordered_correctly.append(eig_sorted.item())
            eigh_ordered_correctly.append(eigh_sorted.item())

            # Sort both sets of eigenvalues for comparison
            eig_sorted_vals, eig_sort_indices = torch.sort(eig_eigenvals_real)
            eigh_sorted_vals = eigh_eigenvals  # eigh already returns sorted eigenvalues

            # Compare eigenvalue differences
            eigenval_diff = torch.max(
                torch.abs(eig_sorted_vals - eigh_sorted_vals)
            ).item()
            eigenval_diffs.append(eigenval_diff)

            # Compare corresponding eigenvectors (need to sort eig eigenvectors)
            eig_sorted_vecs = eig_eigenvecs.real[:, eig_sort_indices]
            eigh_vecs = eigh_eigenvecs

            # Eigenvectors can differ by sign, so check both orientations
            vec_diff_positive = torch.mean(
                torch.abs(eig_sorted_vecs - eigh_vecs)
            ).item()
            vec_diff_negative = torch.mean(
                torch.abs(eig_sorted_vecs + eigh_vecs)
            ).item()
            vec_diff = min(vec_diff_positive, vec_diff_negative)
            eigenvec_diffs.append(vec_diff)

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue

    # ---- Print Summary Statistics ----
    print(f"\n=== Analysis Results for {len(eigenval_diffs)} samples ===")

    print(f"\n--- Eigenvalue Ordering ---")
    eig_ordered_pct = np.mean(eig_ordered_correctly) * 100
    eigh_ordered_pct = np.mean(eigh_ordered_correctly) * 100
    print(
        f"torch.linalg.eig eigenvalues ordered (smallest first): {eig_ordered_pct:.1f}% of samples"
    )
    print(
        f"torch.linalg.eigh eigenvalues ordered (smallest first): {eigh_ordered_pct:.1f}% of samples"
    )

    print(f"\n--- Hessian Asymmetry Errors ---")
    print(f"Asymmetry error - Max: {np.max(asymmetry_errors):.2e}")
    print(f"Asymmetry error - Min: {np.min(asymmetry_errors):.2e}")
    print(f"Asymmetry error - Avg: {np.mean(asymmetry_errors):.2e}")
    print(f"Asymmetry error - Std: {np.std(asymmetry_errors):.2e}")

    print(f"\n--- Eigenvalue Differences (eig vs eigh) ---")
    print(f"Max eigenvalue difference: {np.max(eigenval_diffs):.2e}")
    print(f"Min eigenvalue difference: {np.min(eigenval_diffs):.2e}")
    print(f"Avg eigenvalue difference: {np.mean(eigenval_diffs):.2e}")
    print(f"Std eigenvalue difference: {np.std(eigenval_diffs):.2e}")

    print(f"\n--- Eigenvector Differences (eig vs eigh) ---")
    print(f"Max eigenvector difference: {np.max(eigenvec_diffs):.2e}")
    print(f"Min eigenvector difference: {np.min(eigenvec_diffs):.2e}")
    print(f"Avg eigenvector difference: {np.mean(eigenvec_diffs):.2e}")
    print(f"Std eigenvector difference: {np.std(eigenvec_diffs):.2e}")

    # ---- Additional Analysis ----
    print(f"\n--- Additional Analysis ---")

    # Analysis for torch.linalg.eigh - smallest eigenvalues only
    print(f"\n[torch.linalg.eigh - Smallest Eigenvalues]")

    smallest_eigenvals_eigh = [eigenvals[0] for eigenvals in eigh_eigenvals_all]
    second_smallest_eigenvals_eigh = [eigenvals[1] for eigenvals in eigh_eigenvals_all]

    print(f"Smallest eigenvalue - Min: {np.min(smallest_eigenvals_eigh):.2e}")
    print(f"Smallest eigenvalue - Max: {np.max(smallest_eigenvals_eigh):.2e}")
    print(f"Smallest eigenvalue - Avg: {np.mean(smallest_eigenvals_eigh):.2e}")

    positive_smallest_eigh = np.sum(np.array(smallest_eigenvals_eigh) > 0)
    positive_second_smallest_eigh = np.sum(np.array(second_smallest_eigenvals_eigh) > 0)
    total_samples = len(smallest_eigenvals_eigh)
    print(
        f"Positive smallest eigenvalues: {positive_smallest_eigh}/{total_samples} ({positive_smallest_eigh/total_samples*100:.1f}%)"
    )
    print(
        f"Positive second smallest eigenvalues: {positive_second_smallest_eigh}/{total_samples} ({positive_second_smallest_eigh/total_samples*100:.1f}%)"
    )

    # Analysis for torch.linalg.eig - smallest eigenvalues only
    print(f"\n[torch.linalg.eig - Smallest Eigenvalues]")

    smallest_eigenvals_eig = [
        np.sort(eigenvals.real)[0] for eigenvals in eig_eigenvals_all
    ]
    second_smallest_eigenvals_eig = [
        np.sort(eigenvals.real)[1] for eigenvals in eig_eigenvals_all
    ]

    print(f"Smallest eigenvalue - Min: {np.min(smallest_eigenvals_eig):.2e}")
    print(f"Smallest eigenvalue - Max: {np.max(smallest_eigenvals_eig):.2e}")
    print(f"Smallest eigenvalue - Avg: {np.mean(smallest_eigenvals_eig):.2e}")

    positive_smallest_eig = np.sum(np.array(smallest_eigenvals_eig) > 0)
    positive_second_smallest_eig = np.sum(np.array(second_smallest_eigenvals_eig) > 0)
    print(
        f"Positive smallest eigenvalues: {positive_smallest_eig}/{total_samples} ({positive_smallest_eig/total_samples*100:.1f}%)"
    )
    print(
        f"Positive second smallest eigenvalues: {positive_second_smallest_eig}/{total_samples} ({positive_second_smallest_eig/total_samples*100:.1f}%)"
    )

    # ---- Critical Point Classification ----
    print(f"\n--- Critical Point Classification ---")

    # Classification for torch.linalg.eigh
    print(f"\n[torch.linalg.eigh - Critical Point Types]")

    local_minima_eigh = 0
    local_maxima_eigh = 0
    index1_saddles_eigh = 0
    higher_saddles_eigh = 0

    for eigenvals in eigh_eigenvals_all:
        negative_count = np.sum(eigenvals < 0)
        if negative_count == 0:
            local_minima_eigh += 1
        elif negative_count == len(eigenvals):
            local_maxima_eigh += 1
        elif negative_count == 1:
            index1_saddles_eigh += 1
        else:
            higher_saddles_eigh += 1

    print(
        f"Local minima (all eigenvalues > 0): {local_minima_eigh}/{total_samples} ({local_minima_eigh/total_samples*100:.1f}%)"
    )
    print(
        f"Local maxima (all eigenvalues < 0): {local_maxima_eigh}/{total_samples} ({local_maxima_eigh/total_samples*100:.1f}%)"
    )
    print(
        f"Index-1 saddle points (1 negative eigenvalue): {index1_saddles_eigh}/{total_samples} ({index1_saddles_eigh/total_samples*100:.1f}%)"
    )
    print(
        f"Higher-order saddle points (>1 negative eigenvalues): {higher_saddles_eigh}/{total_samples} ({higher_saddles_eigh/total_samples*100:.1f}%)"
    )

    # Classification for torch.linalg.eig
    print(f"\n[torch.linalg.eig - Critical Point Types]")

    local_minima_eig = 0
    local_maxima_eig = 0
    index1_saddles_eig = 0
    higher_saddles_eig = 0

    for eigenvals in eig_eigenvals_all:
        eigenvals_real = eigenvals.real
        negative_count = np.sum(eigenvals_real < 0)
        if negative_count == 0:
            local_minima_eig += 1
        elif negative_count == len(eigenvals_real):
            local_maxima_eig += 1
        elif negative_count == 1:
            index1_saddles_eig += 1
        else:
            higher_saddles_eig += 1

    print(
        f"Local minima (all eigenvalues > 0): {local_minima_eig}/{total_samples} ({local_minima_eig/total_samples*100:.1f}%)"
    )
    print(
        f"Local maxima (all eigenvalues < 0): {local_maxima_eig}/{total_samples} ({local_maxima_eig/total_samples*100:.1f}%)"
    )
    print(
        f"Index-1 saddle points (1 negative eigenvalue): {index1_saddles_eig}/{total_samples} ({index1_saddles_eig/total_samples*100:.1f}%)"
    )
    print(
        f"Higher-order saddle points (>1 negative eigenvalues): {higher_saddles_eig}/{total_samples} ({higher_saddles_eig/total_samples*100:.1f}%)"
    )

    # ---- Timing ----
    print(f"\n--- Timing ---")
    t0 = time.time()
    for sample_idx in range(min(num_samples, len(dataset))):
        # Get sample and extract hessian
        sample = dataset[sample_idx]
        dft_hessian = sample.hessian
        n_atoms = sample.pos.shape[0]
        # Reshape hessian to [3*N, 3*N] format
        hessian_matrix = dft_hessian.reshape(n_atoms * 3, n_atoms * 3)
        eig_eigenvals, eig_eigenvecs = torch.linalg.eig(hessian_matrix)
    eig_elapsed = time.time() - t0
    print(f"Timing: 100 samples with torch.linalg.eig: {eig_elapsed:.4f} seconds")

    t0 = time.time()
    for sample_idx in range(min(num_samples, len(dataset))):
        # Get sample and extract hessian
        sample = dataset[sample_idx]
        dft_hessian = sample.hessian
        n_atoms = sample.pos.shape[0]
        # Reshape hessian to [3*N, 3*N] format
        hessian_matrix = dft_hessian.reshape(n_atoms * 3, n_atoms * 3)
        eigh_eigenvals, eigh_eigenvecs = torch.linalg.eigh(hessian_matrix)
    eigh_elapsed = time.time() - t0
    print(f"Timing: 100 samples with torch.linalg.eigh: {eigh_elapsed:.4f} seconds")

    return {
        "asymmetry_errors": asymmetry_errors,
        "eigenval_diffs": eigenval_diffs,
        "eigenvec_diffs": eigenvec_diffs,
        "eig_ordered_correctly": eig_ordered_correctly,
        "eigh_ordered_correctly": eigh_ordered_correctly,
        "eigenvals_eig": eig_eigenvals_all,
        "eigenvals_eigh": eigh_eigenvals_all,
    }


if __name__ == "__main__":
    """
    python playground/inspect_dft_hess_eigen_stats.py
    python playground/inspect_dft_hess_eigen_stats.py --dataset-file RGD1.lmdb --num-samples 50 --flip-sign
    """
    parser = argparse.ArgumentParser(
        description="Inspect DFT Hessian eigenvalue computations"
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="ts1x-val.lmdb",
        help="Name of the dataset file to process (default: ts1x-val.lmdb)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to analyze (default: 1000)",
    )
    parser.add_argument(
        "--flip-sign", action="store_true", help="Flip the sign of the Hessian"
    )
    args = parser.parse_args()

    inspect_dft_hess_eigen(
        dataset_file=args.dataset_file,
        num_samples=args.num_samples,
        flip_sign=args.flip_sign,
    )
