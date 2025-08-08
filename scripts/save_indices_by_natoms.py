#!/usr/bin/env python3
"""
Script to load a dataset and save the indices of samples grouped by the number of atoms.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
import json

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path
from gadff.horm.eval_speed import save_idx_by_natoms


if __name__ == "__main__":
    """
    python scripts/save_indices_by_natoms.py ts1x-val.lmdb
    python scripts/save_indices_by_natoms.py ts1x_hess_train_big.lmdb
    python scripts/save_indices_by_natoms.py RGD1.lmdb
    """
    parser = argparse.ArgumentParser(
        description="Save dataset indices grouped by the number of atoms."
    )
    # ts1x-val.lmdb ts1x_hess_train_big.lmdb
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the LMDB dataset file.",
        default="ts1x-val.lmdb",
    )
    args = parser.parse_args()
    results = save_idx_by_natoms(args)

    with open(results["all_path"], "r") as f:
        lines = f.readlines()
        print("First 10 lines of the output JSON file:")
        for line in lines[:10]:
            print(line.rstrip())
