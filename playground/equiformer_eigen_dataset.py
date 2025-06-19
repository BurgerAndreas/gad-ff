#!/usr/bin/env python3
"""
Script to create new LMDB datasets with all original fields plus the smallest two eigenvalues and eigenvectors of the Hessian (from EquiformerV2).
Processes all three datasets:
- RGD1.lmdb
- ts1x_hess_train_big.lmdb
- ts1x-val.lmdb
For each, creates a -eigen.lmdb file with the new fields.
"""
import os
import pickle
import lmdb
import torch
from tqdm import tqdm
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.horm.hessian_utils import compute_hessian_batches, get_smallest_eigenvec_and_values_from_batched_hessians
from gadff.dirutils import find_project_root

def create_eigen_dataset():
    # ---- Config ----
    root_dir = find_project_root()
    dataset_dir = os.path.expanduser(
        "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
    )
    dataset_files = [
        "RGD1.lmdb",
        "ts1x_hess_train_big.lmdb",
        "ts1x-val.lmdb",
    ]
    checkpoint_path = os.path.join(root_dir, "ckpt/eqv2.ckpt")
    batch_size = 1  # must be 1 for per-sample Hessian
    
    # If to save the Hessian of the original dataset again
    # It is by far the largest part of the dataset
    save_hessian = False

    # ---- Load model once ----
    print(f"Loading model from: {checkpoint_path}")
    model = PotentialModule.load_from_checkpoint(checkpoint_path, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    summary = []

    for dataset_file in dataset_files:
        input_lmdb_path = os.path.join(dataset_dir, dataset_file)
        output_lmdb_path = input_lmdb_path.replace(".lmdb", "-eigen.lmdb")
        print(f"\nProcessing {input_lmdb_path} -> {output_lmdb_path}")
        # ---- Load dataset ----
        dataset = LmdbDataset(input_lmdb_path)
        print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")
        
        # ---- Print keys of first sample ----
        first_sample = dataset[0]
        print("Keys in first sample:", list(first_sample.keys()))
        print("Shapes per key:")
        for key in first_sample.keys():
            print(f"{key}: {first_sample[key].shape}")
        
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # ---- Prepare output LMDB ----
        map_size = 2 * os.path.getsize(input_lmdb_path)  # generous size
        out_env = lmdb.open(output_lmdb_path, map_size=map_size)
        
        # ---- Main loop ----
        with out_env.begin(write=True) as txn:
            for idx in tqdm(range(len(dataset)), desc=f"Processing {dataset_file}"):
                data = dataset[idx]
                # Move to device and prepare
                data = data.to(device)
                data.pos.requires_grad_(True)
                data = compute_extra_props(data)
                # Forward pass
                with torch.no_grad():
                    energy, forces = model.potential.forward(data)
                # Compute Hessian and eigenpairs
                hessians = compute_hessian_batches(data, data.pos, energy, forces)
                smallest_eigenvals, smallest_eigenvecs = get_smallest_eigenvec_and_values_from_batched_hessians(
                    data, hessians, n_smallest=2
                )
                # Flatten eigenvectors to shape [2, N_atoms*3]
                n_atoms = data.natoms.item() if hasattr(data.natoms, 'item') else int(data.natoms)
                eigvecs = smallest_eigenvecs[0].T.contiguous().reshape(2, n_atoms*3).cpu()
                eigvals = smallest_eigenvals[0].cpu()
                # Add new fields
                data.hessian_eigenvalues = eigvals
                data.hessian_eigenvectors = eigvecs
                if not save_hessian:
                    # remove data.hessian
                    data.hessian = None
                # Serialize and write
                txn.put(f"{idx}".encode("ascii"), pickle.dumps(data))
            # Store length
            txn.put("length".encode("ascii"), pickle.dumps(len(dataset)))
        print(f"Done. New dataset written to {output_lmdb_path}")
        summary.append((dataset_file, len(dataset), output_lmdb_path))

    print("\nAll datasets processed.")
    for fname, n, outpath in summary:
        print(f"{fname}: {n} samples -> {outpath}") 
        
if __name__ == "__main__":
    create_eigen_dataset()