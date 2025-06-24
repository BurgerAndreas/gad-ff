#!/usr/bin/env python3
"""
Script to create new LMDB datasets with all original fields plus the smallest two eigenvalues and eigenvectors of the DFT-computed Hessian.

Creates a modified version of the dataset, which also includes the smallest two eigenvectors and eigenvalues of the 
DFT-computed Hessian (already saved in the .hessian field). 
The script loops through the dataset, extracts the eigenvalues and eigenvectors from the existing Hessian, and saves the predictions. 
Saves a new dataset instead of appending new columns to the existing dataset.
Uses the same format as the original dataset. 
Adds all keys of the old dataset to the new dataset. 
Ensures that the ordering (indexing) between the old and the new dataset is also the same.

Processes all three datasets:
- RGD1.lmdb
- ts1x_hess_train_big.lmdb
- ts1x-val.lmdb
For each, creates a -dfteigen.lmdb file with the new fields.

Saves the new dataset to 
~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/ts1x-val-dfteigen.lmdb

Much faster than computing Hessians since it uses pre-computed DFT Hessians.
"""
import argparse
import os
import pickle
import lmdb
import torch
import copy
from tqdm import tqdm
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.horm.hessian_utils import get_smallest_eigen_from_full_hessian
from gadff.path_config import DATASET_DIR_HORM_EIGEN, _fix_dataset_path, remove_dir_recursively




def create_dfteigen_dataset(save_hessian=False, dataset_file="ts1x-val.lmdb", debug=False):
    """
    Creates a new dataset with the smallest two eigenvalues and eigenvectors of the DFT-computed Hessian.
    Saves the new dataset to a new file.
    Adds all keys of the old dataset to the new dataset.
    Ensures that the ordering (indexing) between the old and the new dataset is also the same.

    Args:
        save_hessian (bool): If True, saves the Hessian of the original dataset again.
            The Hessian is by far the largest part of the dataset.
            If False, removes the Hessian from the original dataset.
    """
    # ---- Config ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary = []

    input_lmdb_path = _fix_dataset_path(dataset_file)
    if debug:
        output_lmdb_path = input_lmdb_path.replace(".lmdb", "-dft-hess-eigen-DEBUG.lmdb")
    else:
        output_lmdb_path = input_lmdb_path.replace(".lmdb", "-dft-hess-eigen.lmdb")
    
    # Clean up old database files if they exist
    successfully_removed = remove_dir_recursively(output_lmdb_path)
    remove_dir_recursively(f"{output_lmdb_path}-lock")
    if not successfully_removed:
        raise RuntimeError(f"Output database file {output_lmdb_path} already exists!")
    
    print(f"\nProcessing {input_lmdb_path} -> {output_lmdb_path}")
    
    # ---- Load dataset ----
    # Load without transform to keep the hessian field
    dataset = LmdbDataset(input_lmdb_path)
    print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")
    
    # ---- Print keys of first sample ----
    first_sample = dataset[0]
    print("Keys in first sample (dataset):", list(first_sample.keys()))
    print("Shapes per key:")
    for key in first_sample.keys():
        print(f"{key}: {first_sample[key].shape}")
    
    # Check if hessian field exists
    if not hasattr(first_sample, 'hessian'):
        raise ValueError("Dataset does not contain 'hessian' field. Cannot compute DFT eigenvalues.")
    
    # ---- Prepare output LMDB ----
    map_size = 2 * os.path.getsize(input_lmdb_path)  # generous size
    out_env = lmdb.open(output_lmdb_path, map_size=map_size, subdir=False)
    
    # ---- Main loop ----
    print("")
    num_samples_written = 0
    with out_env.begin(write=True) as txn:
        for sample_idx in tqdm(range(len(dataset)), total=len(dataset)):
            
            try:
                # Get the original sample
                original_sample = dataset[sample_idx]
                data_copy = copy.deepcopy(original_sample)
                
                # Compute smallest eigenvalues and eigenvectors from DFT Hessian
                dft_hessian = original_sample.hessian  # Shape should be [3*N * 3*N]
            
                # Memory movement overhead is not worth it
                # dft_hessian = dft_hessian.to(device)
                
                n_atoms = original_sample.pos.shape[0] # [N]
                dft_hessian = dft_hessian.reshape(n_atoms*3, n_atoms*3) # [N*3, N*3]
                
                # [N*3], [N*3, N*3]
                eigenvalues, eigenvectors = torch.linalg.eigh(dft_hessian)
                
                # Sort eigenvalues and eigenvectors so that eigenvalues are in ascending order
                sorted_eigenvals, sort_indices = torch.sort(eigenvalues)
                print(f"eigenvalues shape: {eigenvalues.shape}")
                print(f"eigenvectors shape: {eigenvectors.shape}")
                print(f"sort_indices shape: {sort_indices.shape}")
                sorted_eigenvecs = eigenvectors[:, sort_indices]
                print(f"eigenvectors shape: {eigenvectors.shape}")
                assert sorted_eigenvals[0] <= sorted_eigenvals[1]
                eigenvalues = sorted_eigenvals
                eigenvectors = sorted_eigenvecs
                
                exit()
                
                smallest_eigenvals = eigenvalues[:2].cpu() # [2]
                smallest_eigenvecs = eigenvectors[:, :2].cpu() # [3*N, 2]
                
                # Reshape eigenvectors to [2, N, 3] format
                eigvecs_reshaped = smallest_eigenvecs.T.reshape(2, n_atoms, 3)  # [2, N, 3]
                
                # Add new fields to the original data object
                data_copy.hessian_eigenvalue_1 = smallest_eigenvals[0:1]  # Keep as [1] tensor
                data_copy.hessian_eigenvalue_2 = smallest_eigenvals[1:2]  # Keep as [1] tensor
                data_copy.hessian_eigenvector_1 = eigvecs_reshaped[0]  # [N, 3]
                data_copy.hessian_eigenvector_2 = eigvecs_reshaped[1]  # [N, 3]
                
                # Optionally remove the hessian to save space
                if not save_hessian:
                    if hasattr(data_copy, 'hessian'):
                        delattr(data_copy, 'hessian')

                txn.put(
                    f"{sample_idx}".encode("ascii"),
                    pickle.dumps(data_copy, protocol=pickle.HIGHEST_PROTOCOL),
                )
                num_samples_written += 1
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}", flush=True)
                exit()
            
        # end of loop
        txn.put(
            "length".encode("ascii"),
            pickle.dumps(num_samples_written, protocol=pickle.HIGHEST_PROTOCOL),
        )
    out_env.close()
    print(f"Done. New dataset written to {output_lmdb_path}")
    summary.append((dataset_file, len(dataset), output_lmdb_path))

    print("\nDataset processed.")
    for fname, n, outpath in summary:
        print(f"{fname}: {n} samples -> {outpath}") 
    return summary

if __name__ == "__main__":
    """Try:
    python scripts/create_dft_hess_eigen_dataset.py --dataset-file data/sample_100.lmdb
    
    python scripts/create_dft_hess_eigen_dataset.py --dataset-file ts1x-val.lmdb
    python scripts/create_dft_hess_eigen_dataset.py --dataset-file RGD1.lmdb
    python scripts/create_dft_hess_eigen_dataset.py --dataset-file ts1x_hess_train_big.lmdb
    """
    parser = argparse.ArgumentParser(description="Create dft-hess-eigen dataset with DFT Hessian eigenvalues and eigenvectors")
    parser.add_argument(
        "--dataset-file", 
        type=str, 
        default="ts1x-val.lmdb",
        help="Name of the dataset file to process (default: ts1x-val.lmdb)"
    )
    parser.add_argument(
        "--save-hessian",
        action="store_true",
        help="Keep the original hessian field in the output dataset (default: False)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (default: False)"
    )
    args = parser.parse_args()
    
    create_dfteigen_dataset(save_hessian=args.save_hessian, dataset_file=args.dataset_file, debug=args.debug) 