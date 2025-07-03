#!/usr/bin/env python3
"""
Parallel script to create new LMDB datasets with all original fields plus the smallest two eigenvalues and eigenvectors of the Hessian (from EquiformerV2).

Usage:
1. Process subset: python script.py --process --dataset RGD1.lmdb --start-idx 0 --end-idx 10000 --job-id 0
2. Combine results: python script.py --combine --dataset RGD1.lmdb

Creates a modified version of the dataset, which also includes the smallest two eigenvectors and eigenvalues of the Hessian of EquiformerV2.
The scripts loops through the dataset, computes the eigenvalues and eigenvectors, and saves the predictions.
Saves a new dataset extra instead of appending new columns to the exisiting dataset.
Uses the same format as the original dataset.
Adds all keys of the old dataset to the new dataset.
Ensures that the ordering (indexing) between the old and the new dataset is also the same.
"""
import os
import pickle
import lmdb
import torch
import copy
import argparse
import glob
from torch_geometric.loader import DataLoader as TGDataLoader
from tqdm import tqdm
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset, remove_hessian_transform
from gadff.horm.hessian_utils import (
    compute_hessian_batches,
    predict_eigen_from_batch,
    compute_hessian_single_batch,
    get_smallest_eigen_from_batched_hessians,
)
from gadff.path_config import find_project_root


def remove_dir_recursively(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            # Remove all files in the directory before removing the directory itself
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    # Recursively remove subdirectories if any
                    import shutil

                    shutil.rmtree(file_path)
            os.rmdir(path)
        else:
            os.remove(path)
    # success if path does not exist anymore
    return not os.path.exists(path)


root_dir = find_project_root()
dataset_dir = os.path.expanduser(
    "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
)
dataset_files = [
    "ts1x-val.lmdb",  # 50844 samples
    "ts1x_hess_train_big.lmdb",  # 1725362 samples
    "RGD1.lmdb",  # 60000 samples
]
checkpoint_path = os.path.join(root_dir, "ckpt/eqv2.ckpt")


class SubsetDataset:
    """Wrapper to access only a subset of indices from an LMDB dataset"""

    def __init__(self, dataset, start_idx, end_idx):
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = min(end_idx, len(dataset))
        self.indices = list(range(self.start_idx, self.end_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def process_subset(dataset_file, start_idx, end_idx, job_id, save_hessian=False):
    """
    Process a subset of the dataset and save to a temporary LMDB file.

    Args:
        dataset_file (str): Name of the dataset file (e.g., "RGD1.lmdb")
        start_idx (int): Starting index (inclusive)
        end_idx (int): Ending index (exclusive)
        job_id (int): Job identifier for unique temporary file naming
        save_hessian (bool): If True, saves the Hessian of the original dataset again.
    """
    input_lmdb_path = os.path.join(dataset_dir, dataset_file)
    temp_output_path = input_lmdb_path.replace(".lmdb", f"-eigen-temp-{job_id}.lmdb")

    # Clean up old temp files if they exist
    successfully_removed = remove_dir_recursively(temp_output_path)
    remove_dir_recursively(f"{temp_output_path}-lock")
    if not successfully_removed:
        raise RuntimeError(
            f"Temp output database file {temp_output_path} already exists!"
        )

    print(
        f"Processing subset {start_idx}:{end_idx} of {input_lmdb_path} -> {temp_output_path}"
    )

    # ---- Load model once ----
    print(f"Loading model from: {checkpoint_path}")
    model = PotentialModule.load_from_checkpoint(checkpoint_path, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # ---- Load dataset ----
    if save_hessian:
        full_dataset = LmdbDataset(input_lmdb_path)
    else:
        full_dataset = LmdbDataset(input_lmdb_path, transform=remove_hessian_transform)

    # Create subset
    subset_dataset = SubsetDataset(full_dataset, start_idx, end_idx)
    print(
        f"Processing subset with {len(subset_dataset)} samples (indices {start_idx}:{end_idx})"
    )

    # ---- Print keys of first sample ----
    first_sample = subset_dataset[0]
    print("Keys in first sample:", list(first_sample.keys()))
    print("Shapes per key:")
    for key in first_sample.keys():
        print(f"{key}: {first_sample[key].shape}")

    dataloader = TGDataLoader(subset_dataset, batch_size=1, shuffle=False)

    # ---- Prepare output LMDB ----
    # Estimate size based on subset size
    full_size = os.path.getsize(input_lmdb_path)
    subset_ratio = len(subset_dataset) / len(full_dataset)
    map_size = int(2 * full_size * subset_ratio) + 1024 * 1024 * 100  # Add 100MB buffer
    out_env = lmdb.open(temp_output_path, map_size=map_size, subdir=False)

    # ---- Main loop ----
    print("")
    processed_samples = []
    with out_env.begin(write=True) as txn:
        for local_idx, batch in tqdm(enumerate(dataloader), total=len(subset_dataset)):
            global_idx = start_idx + local_idx

            # Make a deep copy to avoid modifying the original data object in memory
            data_copy = copy.deepcopy(subset_dataset[local_idx])

            # atomization energy. shape used by equiformerv2
            if not hasattr(batch, "ae"):
                batch.ae = torch.zeros_like(batch.energy)

            # tqdm.write(f"Processing global index {global_idx}, batch size: {batch.pos.shape}")

            # Move to device and prepare
            batch = batch.to(device)
            batch = compute_extra_props(batch)

            # Forward pass
            energy, forces, out = model.potential.forward(batch)

            # Compute Hessian and eigenpairs
            hessians = compute_hessian_batches(batch, batch.pos, None, forces)
            smallest_eigenvals, smallest_eigenvecs = (
                get_smallest_eigen_from_batched_hessians(batch, hessians, n_smallest=2)
            )

            # Flatten eigenvectors to shape [2, N_atoms*3]
            n_atoms = (
                batch.natoms.item()
                if hasattr(batch.natoms, "item")
                else int(batch.natoms)
            )
            eigvecs = smallest_eigenvecs[0].T.contiguous().reshape(2, n_atoms * 3).cpu()
            eigvals = smallest_eigenvals[0].cpu()

            # Add new fields to the original data object
            data_copy.hessian_eigenvalues = eigvals
            data_copy.hessian_eigenvectors = eigvecs

            # Store with global index as key
            txn.put(
                f"{global_idx}".encode("ascii"),
                pickle.dumps(data_copy, protocol=pickle.HIGHEST_PROTOCOL),
            )
            processed_samples.append(global_idx)

        # Store metadata about this subset
        metadata = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "job_id": job_id,
            "processed_samples": processed_samples,
            "length": len(processed_samples),
        }
        txn.put(
            "metadata".encode("ascii"),
            pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL),
        )

    out_env.close()
    print(f"Subset processing complete. Temp file: {temp_output_path}")
    print(
        f"Processed {len(processed_samples)} samples with indices: {min(processed_samples)}-{max(processed_samples)}"
    )
    return temp_output_path, len(processed_samples)


def combine_subsets(dataset_file, save_hessian=False):
    """
    Combine all temporary subset files into the final output dataset.

    Args:
        dataset_file (str): Name of the original dataset file (e.g., "RGD1.lmdb")
        save_hessian (bool): Whether hessian was saved in the subsets
    """
    input_lmdb_path = os.path.join(dataset_dir, dataset_file)
    output_lmdb_path = input_lmdb_path.replace(".lmdb", "-eigen.lmdb")

    # Find all temporary files
    temp_pattern = input_lmdb_path.replace(".lmdb", "-eigen-temp-*.lmdb")
    temp_files = glob.glob(temp_pattern)

    if not temp_files:
        raise RuntimeError(f"No temporary files found matching pattern: {temp_pattern}")

    print(f"Found {len(temp_files)} temporary files to combine")
    for temp_file in sorted(temp_files):
        print(f"  - {temp_file}")

    # Clean up final output file if it exists
    successfully_removed = remove_dir_recursively(output_lmdb_path)
    remove_dir_recursively(f"{output_lmdb_path}-lock")
    if not successfully_removed:
        raise RuntimeError(f"Output database file {output_lmdb_path} already exists!")

    # Get original dataset size to estimate map size
    original_size = os.path.getsize(input_lmdb_path)
    map_size = 3 * original_size  # generous size for combined dataset

    # Open final output database
    out_env = lmdb.open(output_lmdb_path, map_size=map_size)

    total_samples = 0
    all_processed_indices = set()

    print("\nCombining temporary files")
    with out_env.begin(write=True) as out_txn:
        for temp_file in tqdm(sorted(temp_files)):
            temp_env = lmdb.open(temp_file, readonly=True)

            with temp_env.begin() as temp_txn:
                # Read metadata
                metadata_bytes = temp_txn.get("metadata".encode("ascii"))
                if metadata_bytes is None:
                    print(f"Warning: No metadata found in {temp_file}")
                    continue

                metadata = pickle.loads(metadata_bytes)
                print(
                    f"Processing temp file with job_id={metadata['job_id']}, "
                    f"indices {metadata['start_idx']}:{metadata['end_idx']}, "
                    f"length={metadata['length']}"
                )

                # Copy all data entries (skip metadata)
                cursor = temp_txn.cursor()
                for key, value in cursor:
                    key_str = key.decode("ascii")
                    if key_str != "metadata":
                        # Copy to final database
                        out_txn.put(key, value)
                        all_processed_indices.add(int(key_str))
                        total_samples += 1

            temp_env.close()

    # Write final metadata
    with out_env.begin(write=True) as out_txn:
        out_txn.put(
            "length".encode("ascii"),
            pickle.dumps(total_samples, protocol=pickle.HIGHEST_PROTOCOL),
        )

    out_env.close()

    print(f"\nCombining complete!")
    print(f"Final dataset: {output_lmdb_path}")
    print(f"Total samples: {total_samples}")
    print(f"Index range: {min(all_processed_indices)}-{max(all_processed_indices)}")

    # Verify completeness
    expected_indices = set(
        range(min(all_processed_indices), max(all_processed_indices) + 1)
    )
    missing_indices = expected_indices - all_processed_indices
    if missing_indices:
        print(
            f"WARNING: Missing indices: {sorted(list(missing_indices))[:10]}{'...' if len(missing_indices) > 10 else ''}"
        )
        print(f"Total missing: {len(missing_indices)}")
    else:
        print("✓ All expected indices are present")

    # Clean up temporary files
    # cleanup_temp_files(dataset_file)

    return output_lmdb_path, total_samples


def cleanup_temp_files(dataset_file):
    """Remove all temporary files for a given dataset"""
    input_lmdb_path = os.path.join(dataset_dir, dataset_file)
    temp_pattern = input_lmdb_path.replace(".lmdb", "-eigen-temp-*.lmdb")
    temp_files = glob.glob(temp_pattern)

    print(f"\nCleaning up {len(temp_files)} temporary files")
    for temp_file in temp_files:
        try:
            remove_dir_recursively(temp_file)
            remove_dir_recursively(f"{temp_file}-lock")
            print(f"Removed: {temp_file}")
        except Exception as e:
            print(f"Failed to remove {temp_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Hessian eigenvalue processing"
    )
    parser.add_argument(
        "--process", action="store_true", help="Process a subset of the dataset"
    )
    parser.add_argument(
        "--combine", action="store_true", help="Combine processed subsets"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset filename (e.g., RGD1.lmdb)"
    )
    parser.add_argument(
        "--start-idx", type=int, help="Starting index for subset processing"
    )
    parser.add_argument(
        "--end-idx", type=int, help="Ending index for subset processing"
    )
    parser.add_argument(
        "--job-id", type=int, help="Job ID for unique temporary file naming"
    )
    parser.add_argument(
        "--save-hessian", action="store_true", help="Save original Hessian data"
    )
    parser.add_argument(
        "--samples-per-job",
        type=int,
        default=10000,
        help="Samples per job for script generation",
    )

    args = parser.parse_args()

    if args.process:
        if args.start_idx is None or args.end_idx is None or args.job_id is None:
            raise ValueError("--process requires --start-idx, --end-idx, and --job-id")

        temp_file, num_samples = process_subset(
            args.dataset, args.start_idx, args.end_idx, args.job_id, args.save_hessian
        )
        print(f"Subset processing complete: {temp_file} ({num_samples} samples)")

    elif args.combine:
        output_file, total_samples = combine_subsets(args.dataset, args.save_hessian)
        print(f"Combining complete: {output_file} ({total_samples} samples)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
