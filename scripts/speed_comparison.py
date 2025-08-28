import torch
import argparse
from tqdm import tqdm
import wandb
from torch_geometric.loader import DataLoader as TGDataLoader
import pandas as pd
import plotly.graph_objects as go
import time
import sys
import json
from pathlib import Path
import os

from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path, DATASET_FILES_HORM
from ocpmodels.hessian_graph_transform import HessianGraphTransform
from gadff.horm.eval_speed import speed_comparison, plot_speed_comparison


if __name__ == "__main__":
    """
    python scripts/speed_comparison.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt
    python scripts/speed_comparison.py --dataset ts1x-val.lmdb --max_samples_per_n 100
    python scripts/speed_comparison.py --dataset ts1x_hess_train_big.lmdb --max_samples_per_n 1000
    """
    parser = argparse.ArgumentParser(
        description="Speed comparison"
    )

    # Subparser for speed comparison
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/eqv2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name",
    )
    parser.add_argument(
        "--max_samples_per_n",
        type=int,
        default=100,
        help="Maximum number of samples per N atoms to test.",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the speed comparison. If false attempt to load existing results.",
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    redo = args.redo

    output_dir = "./resultseval"
    if not redo:
        output_dir = Path(output_dir)
        output_path = output_dir / f"{args.dataset}_speed_comparison_results.csv"
        if output_path.exists():
            results_df = pd.read_csv(output_path)
            print(f"Loaded existing results from {output_path}")
        else:
            redo = True

    if redo:
        results_df = speed_comparison(
            checkpoint_path=args.ckpt_path,
            dataset_name=args.dataset,
            max_samples_per_n=args.max_samples_per_n,
            output_dir=output_dir,
        )

    # Plot results
    plot_speed_comparison(results_df)

    print("Done.")
