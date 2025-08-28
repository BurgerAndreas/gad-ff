#!/usr/bin/env python3
"""
Test script to verify that the Hessian computation optimization works correctly.
This script compares the old approach vs the new approach to ensure they produce identical results.
"""

import torch
import time
from pathlib import Path
from torch_geometric.data import Batch as TGBatch
from torch_geometric.loader import DataLoader as TGDataLoader

from gadff.horm.ff_lmdb import LmdbDataset
from ocpmodels.hessian_graph_transform import (
    HessianGraphTransform,
    create_hessian_collate_fn,
    HessianDataLoader,
)
from nets.equiformer_v2.hessian_pred_utils import add_extra_props_for_hessian


def test_hessian_optimization():
    """Test that the new dataloader approach produces identical results to the old approach."""

    # Find a test dataset
    test_data_path = Path("data/sample_100.lmdb")  # Adjust path as needed
    if not test_data_path.exists():
        print(f"Test dataset not found at {test_data_path}")
        print("Please provide a valid LMDB dataset path to test")
        return

    # Create transform
    transform = HessianGraphTransform(cutoff=100.0, cutoff_hessian=100.0, max_neighbors=None, use_pbc=False)

    # Load dataset with transform
    dataset = LmdbDataset(test_data_path, transform=transform)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Test with small batch size
    batch_size = 2
    follow_batch = ["diag_ij", "edge_index", "message_idx_ij"]

    # Create dataloaders
    # 1. Old approach: standard collate, then add_extra_props_for_hessian in training loop
    dataloader_old = TGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=0,  # Use 0 for deterministic testing
    )

    # 2. New approach: custom collate function that applies offsetting
    dataloader_new = HessianDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=0,  # Use 0 for deterministic testing
        do_hessian_batch_offsetting=True,
    )

    print(f"Testing with batch size {batch_size}")

    # Compare first batch
    batch_old = next(iter(dataloader_old))
    batch_new = next(iter(dataloader_new))

    print(
        f"Batch properties - Old: batch={batch_old.batch.max().item() + 1}, atoms={batch_old.natoms.sum()}"
    )
    print(
        f"Batch properties - New: batch={batch_new.batch.max().item() + 1}, atoms={batch_new.natoms.sum()}"
    )

    # Apply old approach
    print("\nTiming old approach...")
    start_time = time.time()
    batch_old_processed = add_extra_props_for_hessian(batch_old, offset_indices=True)
    old_time = time.time() - start_time
    print(f"Old approach time: {old_time * 1000:.2f}ms")

    # New approach is already processed
    print(f"New approach: batch already processed in dataloader worker")

    # Compare key attributes
    attrs_to_compare = [
        "ptr_1d_hessian",
        "message_idx_ij",
        "message_idx_ji",
        "diag_ij",
        "diag_ji",
        "node_transpose_idx",
    ]

    print(f"\nComparing batch attributes...")
    all_match = True
    for attr in attrs_to_compare:
        if hasattr(batch_old_processed, attr) and hasattr(batch_new, attr):
            old_val = getattr(batch_old_processed, attr)
            new_val = getattr(batch_new, attr)
            if torch.allclose(old_val, new_val):
                print(f"✓ {attr}: MATCH")
            else:
                print(f"✗ {attr}: MISMATCH")
                print(
                    f"  Old shape: {old_val.shape if hasattr(old_val, 'shape') else 'scalar'}"
                )
                print(
                    f"  New shape: {new_val.shape if hasattr(new_val, 'shape') else 'scalar'}"
                )
                print(
                    f"  Max diff: {torch.max(torch.abs(old_val - new_val)).item():.2e}"
                )
                all_match = False
        elif hasattr(batch_old_processed, attr):
            print(f"✗ {attr}: Missing in new batch")
            all_match = False
        elif hasattr(batch_new, attr):
            print(f"✗ {attr}: Missing in old batch")
            all_match = False
        else:
            print(f"- {attr}: Not present in either")

    # Check offsetdone flag
    if hasattr(batch_new, "offsetdone") and batch_new.offsetdone:
        print("✓ offsetdone flag set correctly in new batch")
    else:
        print("✗ offsetdone flag not set in new batch")
        all_match = False

    if all_match:
        print(f"\n🎉 SUCCESS: Both approaches produce identical results!")
        print(
            f"   Computation moved to dataloader workers can hide {old_time * 1000:.2f}ms latency"
        )
    else:
        print(f"\n❌ FAILURE: Results don't match - check implementation")

    return all_match


if __name__ == "__main__":
    test_hessian_optimization()
