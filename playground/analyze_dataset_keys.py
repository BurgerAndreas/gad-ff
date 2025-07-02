#!/usr/bin/env python3
"""
Script to analyze dataset schemas and identify differences between datasets.
This helps understand why ConcatDataset fails and what attributes need to be unified.
"""

import sys
import os
from pathlib import Path
import torch

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import find_project_root, _fix_dataset_path
from ocpmodels.units import ATOMIC_NUMBER_TO_ELEMENT
from nets.prediction_utils import GLOBAL_ATOM_NUMBERS


def analyze_sample_schema(sample, dataset_name):
    """Analyze a single sample and return its schema information."""
    print(f"\n=== Dataset: {dataset_name} ===")

    # Get all attributes
    attrs = [
        attr
        for attr in dir(sample)
        if not attr.startswith("_") and not callable(getattr(sample, attr))
    ]

    print(f"Total attributes: {len(attrs)}")
    print(f"Attributes: {sorted(attrs)}")

    # Analyze each attribute
    attr_info = {}
    for attr in sorted(attrs):
        try:
            value = getattr(sample, attr)
            if isinstance(value, torch.Tensor):
                info = f"torch.Tensor{tuple(value.shape)} dtype={value.dtype}"
            elif isinstance(value, (int, float)):
                info = f"{type(value).__name__}: {value}"
            elif isinstance(value, str):
                info = f"str: '{value}'"
            else:
                info = f"{type(value).__name__}: {str(value)[:50]}"

            attr_info[attr] = info
            print(f"  {attr:20} -> {info}")

        except Exception as e:
            print(f"  {attr:20} -> ERROR: {e}")
            attr_info[attr] = f"ERROR: {e}"

    return set(attrs), attr_info


def main():
    print("Dataset Schema Analysis")
    print("=" * 50)

    # Define the datasets to analyze
    datasets_to_check = [
        "ts1x_hess_train_big-dft-hess-eigen.lmdb",
        "RGD1-dft-hess-eigen.lmdb",
        "ts1x-val-dft-hess-eigen.lmdb",
        # Add any other datasets you want to check
        "data/sample_100.lmdb",
    ]

    for dataset_path in datasets_to_check:
        try:
            print(f"\nLoading dataset: {dataset_path}")

            # Fix the path using the same method as training
            fixed_path = _fix_dataset_path(dataset_path)
            print(f"Fixed path: {fixed_path}")

            # Load the dataset
            dataset = LmdbDataset(Path(fixed_path))
            print(f"Dataset length: {len(dataset)}")

            if len(dataset) > 0:
                # Analyze first sample
                sample = dataset[0]
                for k, v in sample.items():
                    print(f"{k}: {type(v)} {v.shape} {v.dtype}")

                # --- Print unique elements in the dataset ---
                all_atomic_numbers = set()
                max_samples = min(100, len(dataset))
                for i in range(max_samples):
                    s = dataset[i]
                    z = getattr(s, "z", None)
                    if z is not None:
                        if isinstance(z, torch.Tensor):
                            all_atomic_numbers.update(z.cpu().numpy().tolist())
                        elif isinstance(z, (list, tuple)):
                            all_atomic_numbers.update(z)
                    elif hasattr(s, "one_hot"):
                        one_hot = getattr(s, "one_hot")
                        if isinstance(one_hot, torch.Tensor):
                            indices = one_hot.long().argmax(dim=1)
                            z_from_one_hot = (
                                GLOBAL_ATOM_NUMBERS[indices].cpu().numpy().tolist()
                            )
                            all_atomic_numbers.update(z_from_one_hot)
                # Remove zeros and non-integer values
                all_atomic_numbers = {int(a) for a in all_atomic_numbers if int(a) > 0}
                elements = sorted(
                    [
                        ATOMIC_NUMBER_TO_ELEMENT.get(a, f"?{a}")
                        for a in all_atomic_numbers
                    ]
                )
                print(f"Unique atomic numbers in {dataset_path}: {all_atomic_numbers}")
                print(f"Unique elements in {dataset_path}: {elements}")
                # --- End unique elements ---
            else:
                print(f"Dataset {dataset_path} is empty!")

        except Exception as e:
            print(f"ERROR loading {dataset_path}: {e}")
            import traceback

            traceback.print_exc()


"""Conclusion
RGD1 lacks:
ae: <class 'torch.Tensor'> torch.Size([]) torch.float32 -> same as energy
rxn: <class 'torch.Tensor'> torch.Size([]) torch.int64 -> add -1 to all

All other (T1x based) datasets lack:
freq: <class 'torch.Tensor'> torch.Size([N*3])
eig_values: <class 'torch.Tensor'> torch.Size([N*3])
force_constant: <class 'torch.Tensor'> torch.Size([N*3])
-> remove these attributes from the dataset
"""

if __name__ == "__main__":
    main()
