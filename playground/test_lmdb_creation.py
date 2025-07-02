#!/usr/bin/env python3
"""
Script to load 10 samples from an LMDB dataset, predict forces using EquiformerV2,
and save the updated data to a new LMDB file.
"""
import os
import pickle
import copy

import lmdb
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as TGDataLoader

from gadff.path_config import find_project_root
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.path_config import (
    DATASET_DIR_HORM_EIGEN,
    DATASET_FILES_HORM,
    DATA_PATH_HORM_SAMPLE,
    CHECKPOINT_PATH_EQUIFORMER_HORM,
)


def load_model(checkpoint_path):
    """Load EquiformerV2 model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    print(f"Model type: {model_name}")
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return model, device


def predict_forces(model, batch, device):
    """Predict forces for a batch."""
    batch = batch.to(device)
    batch.pos.requires_grad_(True)
    batch = compute_extra_props(batch)
    model_name = model.model_config["name"]
    if model_name == "LEFTNet":
        _, forces = model.potential.forward_autograd(batch)
    else:
        _, forces = model.potential.forward(batch)
    return forces


def main():
    """Main function to run prediction and create new dataset."""
    root_dir = find_project_root()
    print(f"Root directory: {root_dir}")
    checkpoint_path = CHECKPOINT_PATH_EQUIFORMER_HORM
    dataset_path = DATA_PATH_HORM_SAMPLE

    output_lmdb_path = os.path.join(
        os.path.dirname(DATA_PATH_HORM_SAMPLE), "test_10_samples_with_pred.lmdb"
    )
    # Clean up old database files if they exist
    if os.path.exists(output_lmdb_path):
        os.remove(output_lmdb_path)
    if os.path.exists(f"{output_lmdb_path}-lock"):
        os.remove(f"{output_lmdb_path}-lock")
    if os.path.exists(output_lmdb_path):
        raise RuntimeError(f"Output database file {output_lmdb_path} already exists!")

    db_dir = os.path.dirname(output_lmdb_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    target_env = lmdb.open(output_lmdb_path, map_size=1099511627776, subdir=False)

    model, device = load_model(checkpoint_path)
    dataset = LmdbDataset(dataset_path)
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    print("\n" + "=" * 50)
    print("Processing samples and writing to LMDB")
    print("=" * 50)

    num_samples_written = 0
    with target_env.begin(write=True) as txn:
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            print(f"Processing and writing sample {i+1}")
            data_object = copy.deepcopy(batch.to_data_list()[0])

            if i == 0:
                print("batch", type(batch))
                print("batch.to_data_list()", type(batch.to_data_list()))
                print("batch.to_data_list()[0]", type(batch.to_data_list()[0]))
                sample = dataset[i]
                print("sample", type(sample))

            forces = predict_forces(model, batch, device)
            data_object.forces_pred = forces.cpu().detach()

            # Move all tensors in the data object to CPU before saving
            data_object = data_object.to("cpu")

            txn.put(
                f"{i}".encode("ascii"),
                pickle.dumps(data_object, protocol=pickle.HIGHEST_PROTOCOL),
            )
            num_samples_written += 1

        txn.put(
            "length".encode("ascii"),
            pickle.dumps(num_samples_written, protocol=pickle.HIGHEST_PROTOCOL),
        )
    target_env.close()
    print(f"\nSuccessfully created LMDB dataset at: {output_lmdb_path}")


def verify_lmdb_creation():
    """
    Script to verify the contents of the newly created LMDB file against the original.
    """
    root_dir = find_project_root()
    print(f"Root directory: {root_dir}")

    # Paths to the datasets
    original_db_path = os.path.join(root_dir, "data/sample_100.lmdb")
    new_db_path = os.path.join(root_dir, "data/test_10_samples_with_pred.lmdb")

    if not os.path.exists(original_db_path) or not os.path.exists(new_db_path):
        print(
            "One or both dataset files not found. Please run the creation script first."
        )
        return

    # Load both datasets
    print(f"Loading original dataset from: {original_db_path}")
    original_dataset = LmdbDataset(original_db_path)
    print(f"Loading new dataset from: {new_db_path}")
    new_dataset = LmdbDataset(new_db_path)

    # Basic length check
    if len(new_dataset) != 10:
        print(f"Error: New dataset has {len(new_dataset)} entries, expected 10.")
        return

    print("\n" + "=" * 50)
    print("Verifying data consistency for the first 10 samples")
    print("=" * 50)

    # Check other attributes for equality
    attrs_to_check = [
        k for k in original_dataset[0].keys() if k in new_dataset[0].keys()
    ]
    print("attrs_to_check", attrs_to_check)

    all_good = True
    for i in range(10):
        print(f"Comparing sample {i}")
        original_data = original_dataset[i]
        new_data = new_dataset[i]

        # Check that 'forces_pred' exists in new data and not in old
        if not hasattr(new_data, "forces_pred"):
            print(f"  [FAIL] Sample {i} in new dataset is missing 'forces_pred'.")
            all_good = False
            continue
        if hasattr(original_data, "forces_pred"):
            print(f"  [FAIL] Sample {i} in original dataset has 'forces_pred'.")
            all_good = False
            continue

        for attr in attrs_to_check:
            if not hasattr(original_data, attr) or not hasattr(new_data, attr):
                print(f"  [FAIL] Attribute '{attr}' missing in one of the samples.")
                all_good = False
                continue

            val_orig = getattr(original_data, attr)
            val_new = getattr(new_data, attr)

            if isinstance(val_orig, torch.Tensor):
                if not torch.allclose(val_orig, val_new):
                    print(f"  [FAIL] Tensor attribute '{attr}' does not match.")
                    all_good = False
            elif val_orig != val_new:
                print(
                    f"  [FAIL] Attribute '{attr}' does not match: {val_orig} vs {val_new}"
                )
                all_good = False

    print("\n" + "=" * 50)
    if all_good:
        print("Verification successful: All checked attributes match!")
    else:
        print("Verification failed: Some attributes did not match.")
    print("=" * 50)


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
    verify_lmdb_creation()
