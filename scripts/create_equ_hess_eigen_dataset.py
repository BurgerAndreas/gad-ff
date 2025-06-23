#!/usr/bin/env python3
"""
Script to create new LMDB datasets with all original fields plus the smallest two eigenvalues and eigenvectors of the Hessian (from EquiformerV2).

Creates a modified version of the dataset, which also includes the smallest two eigenvectors and eigenvalues of the Hessian of EquiformerV2. 
The scripts loops through the dataset, computes the eigenvalues and eigenvectors, and saves the predictions. 
Saves a new dataset extra instead of appending new columns to the exisiting dataset.
Uses the same format as the original dataset. 
Adds all keys of the old dataset to the new dataset. 
Ensures that the ordering (indexing) between the old and the new dataset is also the same.

Processes all three datasets:
- RGD1.lmdb
- ts1x_hess_train_big.lmdb
- ts1x-val.lmdb
For each, creates a -equ-hess-eigen.lmdb file with the new fields.

Saves the new dataset to 
~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/ts1x-val-equ-hess-eigen.lmdb

Quite slow, takes ~10h for 50k samples, and ~2 weeks for 1.7M samples.
"""
import argparse
import os
import pickle
import lmdb
import torch
import copy
from torch_geometric.loader import DataLoader as TGDataLoader
from tqdm import tqdm
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset, remove_hessian_transform
from gadff.horm.hessian_utils import compute_hessian_batches, predict_eigen_from_batch, compute_hessian_single_batch, get_smallest_eigen_from_batched_hessians
from gadff.path_config import DATASET_DIR_HORM_EIGEN, CHECKPOINT_PATH_EQUIFORMER_HORM

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


def create_eigen_dataset(save_hessian=False, dataset_file="ts1x-val.lmdb"):
    """
    Creates a new dataset with the smallest two eigenvalues and eigenvectors of the Hessian of EquiformerV2.
    Saves the new dataset to a new file.
    Adds all keys of the old dataset to the new dataset.
    Ensures that the ordering (indexing) between the old and the new dataset is also the same.

    Args:
        save_hessian (bool): If True, saves the Hessian of the original dataset again.
            The Hessian is by far the largest part of the dataset.
            If False, removes the Hessian from the original dataset.
    """
    # ---- Config ----
    batch_size = 1  # must be 1 for per-sample Hessian
    
    # ---- Load model once ----
    print(f"Loading model from: {CHECKPOINT_PATH_EQUIFORMER_HORM}")
    model = PotentialModule.load_from_checkpoint(CHECKPOINT_PATH_EQUIFORMER_HORM, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    summary = []

    if os.path.exists(dataset_file):
        input_lmdb_path = os.path.abspath(dataset_file)
    else:
        input_lmdb_path = os.path.join(DATASET_DIR_HORM_EIGEN, dataset_file)
    output_lmdb_path = input_lmdb_path.replace(".lmdb", "-equ-hess-eigen.lmdb")
    # Clean up old database files if they exist
    successfully_removed = remove_dir_recursively(output_lmdb_path)
    remove_dir_recursively(f"{output_lmdb_path}-lock")
    if not successfully_removed:
        raise RuntimeError(f"Output database file {output_lmdb_path} already exists!")
    
    print(f"\nProcessing {input_lmdb_path} -> {output_lmdb_path}")
    # ---- Load dataset ----
    
    if save_hessian:
        dataset = LmdbDataset(input_lmdb_path)
    else:
        dataset = LmdbDataset(input_lmdb_path, transform=remove_hessian_transform)
    print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")
    
    # ---- Print keys of first sample ----
    first_sample = dataset[0]
    print("Keys in first sample (dataset):", list(first_sample.keys()))
    print("Shapes per key:")
    for key in first_sample.keys():
        print(f"{key}: {first_sample[key].shape}")
        
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)
    
    first_batch = next(iter(dataloader))
    print("Keys in first batch (dataloader):", list(first_batch.keys()))
    print("Shapes per key:")
    for key in first_batch.keys():
        print(f"{key}: {first_batch[key].shape}")
    
    # ---- Prepare output LMDB ----
    map_size = 2 * os.path.getsize(input_lmdb_path)  # generous size
    out_env = lmdb.open(output_lmdb_path, map_size=map_size, subdir=False)
    
    # ---- Main loop ----
    print("")
    num_samples_written = 0
    with out_env.begin(write=True) as txn:
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataset)):
            
            # atomization energy. shape used by equiformerv2
            if not hasattr(batch, 'ae'):
                batch.ae = torch.zeros_like(batch.energy)
            
            # tqdm.write(f"Size of batch: {batch.pos.shape}")

            # Move to device and prepare
            batch = batch.to(device)
            # batch.pos.requires_grad_(True)
            batch = compute_extra_props(batch)
            
            # Forward pass
            energy, forces = model.potential.forward(batch)
            
            # Compute Hessian and eigenpairs
            # smallest_eigenvals, smallest_eigenvecs = predict_eigen_from_batch(batch=batch, model=model)
            hessians = compute_hessian_batches(batch, batch.pos, None, forces)
            smallest_eigenvals, smallest_eigenvecs = get_smallest_eigen_from_batched_hessians(batch, hessians, n_smallest=2)
            
            # Flatten eigenvectors to shape [2, N_atoms*3]
            n_atoms = batch.natoms.item() if hasattr(batch.natoms, 'item') else int(batch.natoms)
            eigvecs = smallest_eigenvecs[0].T.contiguous().reshape(2, n_atoms*3).cpu()
            eigvals = smallest_eigenvals[0].cpu()
            
            # Get the original sample from dataset (not from batch) to avoid DataLoader artifacts
            original_sample = dataset[batch_idx]
            data_copy = copy.deepcopy(original_sample)
            
            # Add new fields to the original data object
            data_copy.hessian_eigenvalue_1 = eigvals[0:1]  # Keep as [1] tensor
            data_copy.hessian_eigenvalue_2 = eigvals[1:2]  # Keep as [1] tensor
            data_copy.hessian_eigenvector_1 = eigvecs[0].reshape(n_atoms, 3)  # [N, 3]
            data_copy.hessian_eigenvector_2 = eigvecs[1].reshape(n_atoms, 3)  # [N, 3]

            txn.put(
                f"{batch_idx}".encode("ascii"),
                pickle.dumps(data_copy, protocol=pickle.HIGHEST_PROTOCOL),
            )
            num_samples_written += 1
            
        # end of loop
        txn.put(
            "length".encode("ascii"),
            pickle.dumps(num_samples_written, protocol=pickle.HIGHEST_PROTOCOL),
        )
    out_env.close()
    print(f"Done. New dataset written to {output_lmdb_path}")
    summary.append((dataset_file, len(dataset), output_lmdb_path))

    print("\nAll datasets processed.")
    for fname, n, outpath in summary:
        print(f"{fname}: {n} samples -> {outpath}") 
    return summary

if __name__ == "__main__":
    """Try:
    python scripts/create_equ_hess_eigen_dataset.py --dataset-file data/sample_100.lmdb
    
    python scripts/create_equ_hess_eigen_dataset.py --dataset-file ts1x-val.lmdb
    python scripts/create_equ_hess_eigen_dataset.py --dataset-file RGD1.lmdb
    python scripts/create_equ_hess_eigen_dataset.py --dataset-file ts1x_hess_train_big.lmdb
    """
    parser = argparse.ArgumentParser(description="Create eigen dataset with Hessian eigenvalues and eigenvectors")
    parser.add_argument(
        "--dataset-file", 
        type=str, 
        default="ts1x-val.lmdb",
        help="Name of the dataset file to process (default: ts1x-val.lmdb)"
    )
    args = parser.parse_args()
    
    create_eigen_dataset(dataset_file=args.dataset_file)