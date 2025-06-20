#!/usr/bin/env python3
"""
Script to create new LMDB datasets with all original fields plus the smallest two eigenvalues and eigenvectors of the Hessian (from EquiformerV2).

create a modified version of the dataset, which also includes the smallest two eigenvectors and eigenvalues of the Hessian of EquiformerV2. Make a plan for a script that loops through the dataset, computes the eigenvalues and eigenvectors, and saves the predictions. Think about if it is better to save a new dataset extra or to append new columns to the exisiting dataset

Create a new dataset/file Use the same format as the original dataset. Add all keys of the old dataset to the new dataset. First test which keys are included in the original dataset. Make sure the ordering (indexing) between the old and the new dataset is also the same.

Processes all three datasets:
- RGD1.lmdb
- ts1x_hess_train_big.lmdb
- ts1x-val.lmdb
For each, creates a -eigen.lmdb file with the new fields.

rm -f ~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/ts1x-val-eigen.lmdb
"""
import os
import pickle
import lmdb
import torch
import copy
from torch_geometric.loader import DataLoader as TGDataLoader
import torch.nn.functional as F
from tqdm import tqdm
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset, remove_hessian_transform
from gadff.horm.hessian_utils import compute_hessian_batches, predict_eigen_from_batch, compute_hessian_single_batch, get_smallest_eigen_from_batched_hessians
from gadff.dirutils import find_project_root

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
    # "ts1x-val.lmdb",
    "ts1x_hess_train_big.lmdb",
    "RGD1.lmdb",
]
checkpoint_path = os.path.join(root_dir, "ckpt/eqv2.ckpt")

def create_eigen_dataset():
    # ---- Config ----
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
        print("Keys in first sample:", list(first_sample.keys()))
        print("Shapes per key:")
        for key in first_sample.keys():
            print(f"{key}: {first_sample[key].shape}")
            
        dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)
        
        # ---- Prepare output LMDB ----
        map_size = 2 * os.path.getsize(input_lmdb_path)  # generous size
        out_env = lmdb.open(output_lmdb_path, map_size=map_size)
        
        # ---- Main loop ----
        print("")
        num_samples_written = 0
        with out_env.begin(write=True) as txn:
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataset)):
                # Make a deep copy to avoid modifying the original data object in memory
                data_copy = copy.deepcopy(batch)

                # Move to device and prepare
                batch = batch.to(device)
                batch.pos.requires_grad_(True)
                batch = compute_extra_props(batch)
                
                # Forward pass
                energy, forces = model.potential.forward(batch)
                
                # Compute Hessian and eigenpairs
                smallest_eigenvals, smallest_eigenvecs = predict_eigen_from_batch(batch=batch, model=model)
                
                # Flatten eigenvectors to shape [2, N_atoms*3]
                n_atoms = data_copy.natoms.item() if hasattr(data_copy.natoms, 'item') else int(data_copy.natoms)
                eigvecs = smallest_eigenvecs.T.contiguous().reshape(2, n_atoms*3).cpu()
                eigvals = smallest_eigenvals.cpu()
                
                # Add new fields to the original data object
                data_copy.hessian_eigenvalues = eigvals
                data_copy.hessian_eigenvectors = eigvecs

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

def test_eigen_dataset():
    """
    Load the ts1x-val-eigen.lmdb dataset, test EquiformerV2 forward pass,
    and compare predicted forces to stored forces and first eigenvector.
    """
    # ---- Config ----
    eigen_dataset_path = os.path.join(dataset_dir, "ts1x-val-eigen.lmdb")
    batch_size = 1
    num_test_samples = 20  # Limit for testing
    
    print(f"\nTesting eigen dataset: {eigen_dataset_path}")
    
    # ---- Load model ----
    print(f"Loading model from: {checkpoint_path}")
    model = PotentialModule.load_from_checkpoint(checkpoint_path, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # ---- Load eigen dataset ----
    try:
        eigen_dataset = LmdbDataset(eigen_dataset_path)
        print(f"Loaded eigen dataset with {len(eigen_dataset)} samples")
    except Exception as e:
        print(f"Error loading eigen dataset: {e}")
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
    
    print(f"\nTesting forward pass on {len(test_subset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing batches")):
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
            stored_eigenvecs = batch.hessian_eigenvectors.cpu()  # Shape: [batch_size, 2, natoms*3]
            
            # Convert predicted forces to CPU
            pred_forces_cpu = pred_forces.cpu()
            
            # For each sample in batch
            atom_idx = 0
            for sample_idx in range(batch.batch.max().item() + 1):
                # Get atoms for this sample
                sample_mask = (batch.batch == sample_idx)
                n_atoms = sample_mask.sum().item()
                
                # Extract forces for this sample
                sample_stored_forces = stored_forces[atom_idx:atom_idx + n_atoms].flatten()  # [natoms*3]
                sample_pred_forces = pred_forces_cpu[atom_idx:atom_idx + n_atoms].flatten()  # [natoms*3]
                
                # Extract first eigenvector for this sample
                first_eigenvector = stored_eigenvecs[sample_idx, 0, :n_atoms*3]  # [natoms*3]
                
                # Compute similarities/differences
                # 1. Cosine similarity between predicted and stored forces
                force_cosine_sim = F.cosine_similarity(
                    sample_pred_forces.unsqueeze(0), 
                    sample_stored_forces.unsqueeze(0)
                ).item()
                force_similarities.append(force_cosine_sim)
                
                # 2. Cosine similarity between predicted forces and first eigenvector (should be different)
                eigenvec_cosine_sim = F.cosine_similarity(
                    sample_pred_forces.unsqueeze(0), 
                    first_eigenvector.unsqueeze(0)
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
    print(f"  Range: [{min(eigenvec_differences):.4f}, {max(eigenvec_differences):.4f}]")
    print(f"  Expected: Low similarity (close to 0.0, indicating they are different)")
    
    # ---- Additional statistics ----
    high_force_sim_count = sum(1 for sim in force_similarities if sim > 0.9)
    low_eigenvec_sim_count = sum(1 for sim in eigenvec_differences if sim < 0.1)
    
    print(f"\nStatistics:")
    print(f"  Samples with force similarity > 0.9: {high_force_sim_count}/{len(force_similarities)} ({100*high_force_sim_count/len(force_similarities):.1f}%)")
    print(f"  Samples with |eigenvector similarity| < 0.1: {low_eigenvec_sim_count}/{len(eigenvec_differences)} ({100*low_eigenvec_sim_count/len(eigenvec_differences):.1f}%)")

if __name__ == "__main__":
    create_eigen_dataset()
    test_eigen_dataset()