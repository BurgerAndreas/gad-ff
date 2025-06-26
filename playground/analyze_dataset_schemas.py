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


def analyze_sample_schema(sample, dataset_name):
    """Analyze a single sample and return its schema information."""
    print(f"\n=== Dataset: {dataset_name} ===")
    
    # Get all attributes
    attrs = [attr for attr in dir(sample) if not attr.startswith('_') and not callable(getattr(sample, attr))]
    
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
                info = f"{type(value).__name__}: {str(value)[:50]}..."
            
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
        "data/sample_100.lmdb"
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
                    print(f"{k}: {type(v)} {v.shape}")
            else:
                print(f"Dataset {dataset_path} is empty!")
                
        except Exception as e:
            print(f"ERROR loading {dataset_path}: {e}")
            import traceback
            traceback.print_exc()
    

"""Conclusion
RGD1 lacks:
ae: <class 'torch.Tensor'> torch.Size([]) -> same as energy
rxn: <class 'torch.Tensor'> torch.Size([]) -> add -1 to all

All other (T1x based) datasets lack:
freq: <class 'torch.Tensor'> torch.Size([N*3])
eig_values: <class 'torch.Tensor'> torch.Size([N*3])
force_constant: <class 'torch.Tensor'> torch.Size([N*3])
-> remove these attributes from the dataset
"""

if __name__ == "__main__":
    main() 