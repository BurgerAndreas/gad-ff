import os
import pickle
import lmdb
import torch
import copy
import argparse
from torch_geometric.loader import DataLoader as TGDataLoader
import torch.nn.functional as F
from tqdm import tqdm
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset, fix_hessian_eigen_transform
from gadff.horm.hessian_utils import (
    compute_hessian_batches,
    predict_eigen_from_batch,
    compute_hessian_single_batch,
    get_smallest_eigen_from_batched_hessians,
)
from gadff.path_config import DATASET_DIR_HORM_EIGEN, CHECKPOINT_PATH_EQUIFORMER_HORM


def test_dataset_consistency(
    original_dataset_name="ts1x-val.lmdb",
    eigen_dataset_name="ts1x-val-dft-hess-eigen.lmdb",
    num_samples=10,
):
    """
    Compare the original dataset with the eigen dataset to ensure all original data is preserved
    and only the hessian eigen information is added.
    """
    print(f"\n=== Testing Dataset Consistency ===")

    # Load both datasets
    if os.path.exists(original_dataset_name):
        original_path = os.path.abspath(original_dataset_name)
    else:
        original_path = os.path.join(DATASET_DIR_HORM_EIGEN, original_dataset_name)
    if os.path.exists(eigen_dataset_name):
        eigen_path = os.path.abspath(eigen_dataset_name)
    else:
        eigen_path = os.path.join(DATASET_DIR_HORM_EIGEN, eigen_dataset_name)

    print(f"Loading original dataset: {original_path}")
    try:
        original_dataset = LmdbDataset(original_path)
        print(f"  Loaded {len(original_dataset)} samples")
    except Exception as e:
        print(f"  Error: {e}")
        return

    print(f"Loading eigen dataset: {eigen_path}")
    try:
        eigen_dataset = LmdbDataset(eigen_path, transform=fix_hessian_eigen_transform)
        print(f"  Loaded {len(eigen_dataset)} samples")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # Check dataset sizes match
    if len(original_dataset) != len(eigen_dataset):
        print(
            f"ERROR: Dataset sizes don't match! Original: {len(original_dataset)}, Eigen: {len(eigen_dataset)}"
        )
        return

    print(f"\nComparing first {num_samples} samples...")

    # Expected new fields in eigen dataset
    eigen_fields = {
        "hessian_eigenvalue_1",
        "hessian_eigenvalue_2",
        "hessian_eigenvector_1",
        "hessian_eigenvector_2",
    }

    all_consistent = True

    for i in range(min(num_samples, len(original_dataset))):
        original_sample = original_dataset[i]
        eigen_sample = eigen_dataset[i]

        # Get all keys from both samples
        original_keys = set(original_sample.keys())
        eigen_keys = set(eigen_sample.keys())

        # remove hessian key
        original_keys.remove("hessian")

        # Check that eigen dataset has all original keys plus the new eigen fields
        missing_original_keys = original_keys - eigen_keys
        expected_new_keys = eigen_keys - original_keys

        if missing_original_keys:
            print(
                f"  Sample {i}: ERROR - Missing original keys in eigen dataset: {missing_original_keys}"
            )
            all_consistent = False
            continue

        if expected_new_keys != eigen_fields:
            print(
                f"  Sample {i}: ERROR - Unexpected new keys. Expected: {eigen_fields}, Got: {expected_new_keys}"
            )
            all_consistent = False
            continue

        # Compare all original fields
        sample_consistent = True
        for key in original_keys:
            original_value = getattr(original_sample, key)
            eigen_value = getattr(eigen_sample, key)

            if isinstance(original_value, torch.Tensor):
                if not torch.equal(original_value, eigen_value):
                    print(f"  Sample {i}, key '{key}': ERROR - Tensors don't match!")
                    print(
                        f"    Original shape: {original_value.shape}, Eigen shape: {eigen_value.shape}"
                    )
                    sample_consistent = False
                    all_consistent = False
            else:
                if original_value != eigen_value:
                    print(f"  Sample {i}, key '{key}': ERROR - Values don't match!")
                    print(f"    Original: {original_value}, Eigen: {eigen_value}")
                    sample_consistent = False
                    all_consistent = False

        # Check shapes of new eigen fields
        try:
            eigenval_1 = getattr(eigen_sample, "hessian_eigenvalue_1")
            eigenval_2 = getattr(eigen_sample, "hessian_eigenvalue_2")
            eigenvec_1 = getattr(eigen_sample, "hessian_eigenvector_1")
            eigenvec_2 = getattr(eigen_sample, "hessian_eigenvector_2")

            n_atoms = len(eigen_sample.pos)

            # Check eigenvalue shapes [1]
            if eigenval_1.shape != torch.Size([1]):
                print(
                    f"  Sample {i}: ERROR - hessian_eigenvalue_1 wrong shape: {eigenval_1.shape}, expected [1]"
                )
                sample_consistent = False
                all_consistent = False

            if eigenval_2.shape != torch.Size([1]):
                print(
                    f"  Sample {i}: ERROR - hessian_eigenvalue_2 wrong shape: {eigenval_2.shape}, expected [1]"
                )
                sample_consistent = False
                all_consistent = False

            # Check eigenvector shapes [N, 3]
            expected_eigenvec_shape = torch.Size([n_atoms, 3])
            if eigenvec_1.shape != expected_eigenvec_shape:
                print(
                    f"  Sample {i}: ERROR - hessian_eigenvector_1 wrong shape: {eigenvec_1.shape}, expected {expected_eigenvec_shape}"
                )
                sample_consistent = False
                all_consistent = False

            if eigenvec_2.shape != expected_eigenvec_shape:
                print(
                    f"  Sample {i}: ERROR - hessian_eigenvector_2 wrong shape: {eigenvec_2.shape}, expected {expected_eigenvec_shape}"
                )
                sample_consistent = False
                all_consistent = False

        except AttributeError as e:
            print(f"  Sample {i}: ERROR - Missing eigen field: {e}")
            sample_consistent = False
            all_consistent = False

        if sample_consistent:
            print(f"  Sample {i}: ✓ Consistent")

    print(f"\n=== Consistency Test Results ===")
    if all_consistent:
        print("✓ SUCCESS: All samples are consistent!")
        print("  - All original data preserved")
        print("  - All eigen fields present with correct shapes")
    else:
        print("✗ FAILURE: Some inconsistencies found!")

    return all_consistent


def test_eigen_dataset(dataset_name="ts1x-val-dft-hess-eigen.lmdb", max_batches=-1):
    """
    Load the ts1x-val-dft-hess-eigen.lmdb dataset, test EquiformerV2 forward pass,
    and compare predicted forces to stored forces and first eigenvector.
    """
    # ---- Config ----
    if os.path.exists(dataset_name):
        eigen_dataset_path = os.path.abspath(dataset_name)
    else:
        eigen_dataset_path = os.path.join(DATASET_DIR_HORM_EIGEN, dataset_name)
    batch_size = 2
    num_test_samples = 20  # Limit for testing

    print(f"\nTesting eigen dataset: {eigen_dataset_path}")

    # ---- Load model ----
    print(f"Loading model from: {CHECKPOINT_PATH_EQUIFORMER_HORM}")
    model = PotentialModule.load_from_checkpoint(
        CHECKPOINT_PATH_EQUIFORMER_HORM, strict=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # ---- Load eigen dataset ----
    try:
        eigen_dataset = LmdbDataset(
            eigen_dataset_path, transform=fix_hessian_eigen_transform
        )
        print(f"Loaded eigen dataset with {len(eigen_dataset)} samples")
    except Exception as e:
        print(f"Error loading eigen dataset: {e}")
        # List all files in the current directory
        print(f"\nFiles in dataset directory ({DATASET_DIR_HORM_EIGEN}):")
        try:
            files = os.listdir(DATASET_DIR_HORM_EIGEN)
            for file in sorted(files):
                file_path = os.path.join(DATASET_DIR_HORM_EIGEN, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size} bytes)")
                elif os.path.isdir(file_path):
                    print(f"  {file}/ (directory)")
        except Exception as e:
            print(f"Error listing directory: {e}")
        print("Make sure to run create_eigen_dataset() first!")
        return

    # ---- Create dataloader ----
    # Limit to first few samples for testing
    test_indices = list(range(min(num_test_samples, len(eigen_dataset))))
    test_subset = torch.utils.data.Subset(eigen_dataset, test_indices)
    dataloader = TGDataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # ---- Test forward pass and comparisons ----
    force_similarities = []
    eigenvec_differences = []

    print("\nFirst sample:")
    first_batch = next(iter(dataloader))
    for key in first_batch.keys():
        value = getattr(first_batch, key)
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    print(f"\nTesting forward pass on {len(test_subset)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing batches")):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            # Move to device
            batch = batch.to(device)

            # Ensure pos requires grad for force computation
            batch.pos.requires_grad_(True)

            # Add extra properties needed by model
            batch = compute_extra_props(batch)

            # Forward pass
            pred_energy, pred_forces = model.potential.forward(batch)

            # Get stored forces and first eigenvector
            stored_forces = batch.forces.cpu()  # Shape: [total_atoms, 3]
            stored_eigenvecs = (
                batch.hessian_eigenvector_1.cpu()
            )  # Shape: [total_atoms, 3]

            # Convert predicted forces to CPU
            pred_forces_cpu = pred_forces.cpu()

            # For each sample in batch
            atom_idx = 0
            for sample_idx in range(batch.batch.max().item() + 1):
                # Get atoms for this sample
                sample_mask = batch.batch == sample_idx
                n_atoms = sample_mask.sum().item()

                # Extract forces for this sample
                sample_stored_forces = stored_forces[
                    atom_idx : atom_idx + n_atoms
                ].flatten()  # [natoms*3]
                sample_pred_forces = pred_forces_cpu[
                    atom_idx : atom_idx + n_atoms
                ].flatten()  # [natoms*3]

                # Extract first eigenvector for this sample (flatten to [natoms*3])
                first_eigenvector = stored_eigenvecs[
                    atom_idx : atom_idx + n_atoms
                ].flatten()  # [natoms*3]

                # Compute similarities/differences
                # 1. Cosine similarity between predicted and stored forces
                force_cosine_sim = F.cosine_similarity(
                    sample_pred_forces.unsqueeze(0), sample_stored_forces.unsqueeze(0)
                ).item()
                force_similarities.append(force_cosine_sim)

                # 2. Cosine similarity between predicted forces and first eigenvector (should be different)
                eigenvec_cosine_sim = F.cosine_similarity(
                    sample_pred_forces.unsqueeze(0), first_eigenvector.unsqueeze(0)
                ).item()
                eigenvec_differences.append(abs(eigenvec_cosine_sim))

                atom_idx += n_atoms

    # ---- Print results ----
    print(f"\n=== Results ===")
    print(f"Tested {len(force_similarities)} samples")

    avg_force_similarity = sum(force_similarities) / len(force_similarities)
    avg_eigenvec_difference = sum(eigenvec_differences) / len(eigenvec_differences)

    print(f"\nForce Comparison (predicted vs stored):")
    print(f"  Average cosine similarity: {avg_force_similarity:.4f}")
    print(f"  Range: [{min(force_similarities):.4f}, {max(force_similarities):.4f}]")
    print(f"  Expected: High similarity (close to 1.0)")

    print(f"\nEigenvector Comparison (predicted forces vs first eigenvector):")
    print(f"  Average |cosine similarity|: {avg_eigenvec_difference:.4f}")
    print(
        f"  Range: [{min(eigenvec_differences):.4f}, {max(eigenvec_differences):.4f}]"
    )
    print(f"  Expected: Low similarity (close to 0.0, indicating they are different)")

    # ---- Additional statistics ----
    high_force_sim_count = sum(1 for sim in force_similarities if sim > 0.9)
    low_eigenvec_sim_count = sum(1 for sim in eigenvec_differences if sim < 0.1)

    print(f"\nStatistics:")
    print(
        f"  Samples with force similarity > 0.9: {high_force_sim_count}/{len(force_similarities)} ({100*high_force_sim_count/len(force_similarities):.1f}%)"
    )
    print(
        f"  Samples with |eigenvector similarity| < 0.1: {low_eigenvec_sim_count}/{len(eigenvec_differences)} ({100*low_eigenvec_sim_count/len(eigenvec_differences):.1f}%)"
    )


if __name__ == "__main__":
    """
    python scripts/test_eigen_dataset.py --original-dataset data/sample_100.lmdb
    python scripts/test_eigen_dataset.py --original-dataset RGD1.lmdb
    python scripts/test_eigen_dataset.py --original-dataset ts1x-val.lmdb
    python scripts/test_eigen_dataset.py --original-dataset ts1x_hess_train_big.lmdb
    """
    parser = argparse.ArgumentParser(description="Test eigen datasets")
    parser.add_argument(
        "--original-dataset",
        type=str,
        default="data/sample_100.lmdb",
        help="Path to original dataset file",
    )
    parser.add_argument(
        "--eigen-dataset",
        type=str,
        default=None,
        help="Path to eigen dataset file (default: auto-derived from original)",
    )

    args = parser.parse_args()

    # Auto-derive eigen dataset name if not provided
    if args.eigen_dataset is None:
        # Insert "-dft-hess-eigen" before the file extension
        base, ext = os.path.splitext(args.original_dataset)
        args.eigen_dataset = f"{base}-dft-hess-eigen{ext}"

    # Test consistency between original and eigen datasets
    test_dataset_consistency(
        original_dataset_name=args.original_dataset,
        eigen_dataset_name=args.eigen_dataset,
        num_samples=10,
    )

    # Test eigen dataset functionality
    test_eigen_dataset(dataset_name=args.eigen_dataset, max_batches=10)
