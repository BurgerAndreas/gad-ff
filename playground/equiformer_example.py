#!/usr/bin/env python3
"""
Script to load EquiformerV2 model and predict energy, forces, and Hessian for a batch of samples.

Key output shapes:
- Energy: [B] where B is batch size
- Forces: [B, max_atoms, 3] padded to max molecule size in batch
- Hessian: [B, max_atoms*3, max_atoms*3] padded to max molecule size in batch
- Atom masks: [B, max_atoms] indicating which atoms are real vs padding
"""

import torch
from torch_geometric.loader import DataLoader as TGDataLoader
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset
import os
from gadff.units import ev_angstrom_2_to_hartree_bohr_2
from gadff.path_config import find_project_root
from gadff.horm.hessian_utils import compute_hessian_batches, get_smallest_eigen_from_batched_hessians


def load_model(checkpoint_path):
    """Load EquiformerV2 model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint to get model info
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    print(f"Model type: {model_name}")
    print(f"Checkpoint keys: {ckpt.keys()}")

    # Load the full model
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    )
    # print(f"Model: {model}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Epoch: {model.current_epoch}")
    print(f"Step: {model.global_step}")

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    return model, device


def load_dataset(dataset_path, batch_size=4):
    """Load HORM dataset and create DataLoader."""
    print(f"Loading dataset from: {dataset_path}")

    dataset = LmdbDataset(dataset_path)
    dataloader = TGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded with {len(dataset)} samples")
    return dataloader


def predict_batch(model, batch, device, get_hessian=False):
    """Predict energy, forces, and Hessian for a batch."""
    # Move batch to device
    batch = batch.to(device)

    # Prepare batch with extra properties and gradients
    batch.pos.requires_grad_(True)
    batch = compute_extra_props(batch)

    # Forward pass to get energy and forces
    model_name = model.model_config["name"]

    if model_name == "LEFTNet":
        energy, forces = model.potential.forward_autograd(batch)
    else:
        energy, forces = model.potential.forward(batch)

    return energy, forces



def main():
    """Main function to run prediction."""
    # Paths
    root_dir = find_project_root()
    print(f"Root directory: {root_dir}")
    checkpoint_path = os.path.join(root_dir, "ckpt/eqv2.ckpt")

    # Try different dataset paths in order of preference
    dataset_paths = [
        # "data/sample_100.lmdb",  # Local small dataset for testing
        os.path.expanduser(
            "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/ts1x-val.lmdb"
        ),
        os.path.expanduser(
            "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/RGD1.lmdb"
        ),
    ]

    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        print("No dataset found! Please check dataset paths.")
        return

    # Load model and dataset
    model, device = load_model(checkpoint_path)
    dataloader = load_dataset(dataset_path, batch_size=1)

    print("\n" + "=" * 50)
    print("Running Predictions")
    print("=" * 50)

    # Predict on a few batches
    for i, batch in enumerate(dataloader):
        # keys: ['pos', 'charges', 'hessian', 'batch', 'forces', 'natoms', 'one_hot', 'ptr', 'rxn', 'energy', 'ae']
        if i >= 3:  # Only process first 3 batches
            break

        print(f"\nBatch {i+1}:")
        print(f"  Number of molecules: {batch.batch.max().item() + 1}")
        print(f"  Total atoms: {batch.pos.shape[0]}")

        # Run prediction (gradients needed for Hessian computation)
        try:
            energy, forces = predict_batch(model, batch, device)
            
            hessians = compute_hessian_batches(batch, batch.pos, energy, forces)

            print(f"  Energy shape: {energy.shape}")
            print(f"  Forces shape: {forces.shape}")
            print(f"  Hessian shape: {hessians[0].shape}")

            # Compute the two smallest eigenvalues and corresponding eigenvectors
            smallest_eigenvals, smallest_eigenvecs = get_smallest_eigen_from_batched_hessians(
                batch, hessians, n_smallest=2
            )
            print(f"  Two smallest eigenvalues: {smallest_eigenvals[0]}")
            print(f"  Corresponding eigenvectors shape: {smallest_eigenvecs[0].shape}")

        except Exception as e:
            print(f"  Error during prediction: {e}")
            continue

    print("\nPrediction completed!")


if __name__ == "__main__":
    torch.set_grad_enabled(True)  # Ensure gradients are enabled for Hessian computation
    main()
