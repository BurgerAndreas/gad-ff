#!/usr/bin/env python3
"""
Script to load a dataset and save the indices of samples grouped by the number of atoms.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import os

import torch
from torch.utils.data import Subset
from tqdm import tqdm
import json

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path

import wandb
from torch_geometric.loader import DataLoader as TGDataLoader
import pandas as pd
import plotly.graph_objects as go
import time
import sys
import json

from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path, DATASET_FILES_HORM
from ocpmodels.hessian_graph_transform import HessianGraphTransform

# https://plotly.com/python/templates/
# ['ggplot2', 'seaborn', 'simple_white', 'plotly',
# 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
# 'ygridoff', 'gridon', 'none']
PLOTLY_TEMPLATE = "plotly_white"

def save_idx_by_natoms(args):
    if isinstance(args, str):
        args = argparse.Namespace(dataset_path=args)
    elif isinstance(args, dict):
        args = argparse.Namespace(**args)

    print(f"Loading dataset: {args.dataset_path}")
    fixed_path = fix_dataset_path(args.dataset_path)
    print(f"Fixed path: {fixed_path}")

    dataset = LmdbDataset(Path(fixed_path))
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        print(f"Dataset {args.dataset_path} is empty!")
        return

    indices_by_natoms = defaultdict(list)

    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[i]
        natoms = 0
        if hasattr(sample, "z"):
            natoms = len(sample.z)
        elif hasattr(sample, "pos"):
            natoms = sample.pos.shape[0]
        else:
            print(f"Warning: Could not determine number of atoms for sample {i}")
            continue

        if natoms > 0:
            indices_by_natoms[natoms].append(i)

    # Sort keys
    sorted_indices_by_natoms = {
        k: sorted(v) for k, v in sorted(indices_by_natoms.items())
    }

    # Generate output file path
    dataset_name = Path(fixed_path).stem
    output_filename = f"{dataset_name}_indices_by_natoms.json"
    output_path = Path(fixed_path).parent / output_filename

    print(f"Saving indices to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(sorted_indices_by_natoms, f, indent=2)

    # --- Save a smaller version for quick access ---
    truncated_indices_by_natoms = {
        k: v[:1000] for k, v in sorted_indices_by_natoms.items()
    }
    small_output_filename = f"{dataset_name}_indices_by_natoms_small.json"
    small_output_path = Path(fixed_path).parent / small_output_filename

    print(f"Saving smaller indices file to: {small_output_path}")
    with open(small_output_path, "w") as f:
        json.dump(truncated_indices_by_natoms, f, indent=2)

    print("Done.")
    return {
        "all_idx": sorted_indices_by_natoms,
        "small_idx": truncated_indices_by_natoms,
        "all_path": output_path,
        "small_path": small_output_path,
    }


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    """Helper function to compute derivatives"""
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    """Compute Hessian matrix using autograd."""
    # Compute forces if not given
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)

    # Get number of components (n_atoms * 3)
    n_comp = forces.reshape(-1).shape[0]

    # Initialize hessian
    hess = []
    for f in forces.reshape(-1):
        # Compute second-order derivative for each element
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)

    # Stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def hess2eigenvalues(hess):
    """Convert Hessian to eigenvalues with unit conversion (hartree to eV, bohr to angstrom)"""
    hartree_to_ev = 27.2114
    bohr_to_angstrom = 0.529177
    ev_angstrom_2_to_hartree_bohr_2 = (bohr_to_angstrom**2) / hartree_to_ev

    hess = hess * ev_angstrom_2_to_hartree_bohr_2
    eigen_values, _ = torch.linalg.eigh(hess)
    return eigen_values


def time_hessian_computation(model, batch, hessian_method):
    """Times a single hessian computation and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    do_autograd = hessian_method == "autograd"

    start_event.record()

    if "equiformer" in model.name.lower():
        if do_autograd:
            batch.pos.requires_grad_()
            ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
            hess = compute_hessian(batch.pos, ener, force)
        else:
            with torch.no_grad():
                ener, force, out = model.forward(batch, otf_graph=False, hessian=True)
                hess = out["hessian"]
    else:
        batch.pos.requires_grad_()
        ener, force, out = model.forward(batch)
        hess = compute_hessian(batch.pos, ener, force)

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def speed_comparison(
    checkpoint_path,
    dataset_name,
    max_samples_per_n,
    device="cuda",
    output_dir="./resultseval",
):
    """Compares the speed of autograd vs prediction for Hessian computation."""
    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name

    # Get indices by number of atoms
    fixed_dataset_path = Path(fix_dataset_path(dataset_name))
    indices_file = (
        fixed_dataset_path.parent
        / f"{fixed_dataset_path.stem}_indices_by_natoms_small.json"
    )

    if indices_file.exists():
        print(f"Loading indices from {indices_file}")
        with open(indices_file, "r") as f:
            indices_by_natoms = json.load(f)
    else:
        print(f"Indices file not found. Generating new indices for {dataset_name}")
        results = save_idx_by_natoms({"dataset_path": dataset_name})
        indices_by_natoms = results["small_idx"]

    # Prepare dataset and dataloader
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset = LmdbDataset(fix_dataset_path(dataset_name), transform=transform)
    
    # do a couple of forward passes to warm up the model
    # populate caches, jit, load cuda kernels, and what not
    loader = TGDataLoader(dataset, batch_size=1, shuffle=False)
    for i, sample in enumerate(loader):
        batch = sample.to(device)
        batch = compute_extra_props(batch)
        time_hessian_computation(model, batch, "prediction")
        torch.cuda.empty_cache()
        time_hessian_computation(model, batch, "autograd")
        torch.cuda.empty_cache()
        if i > 10:
            break
    print("Model warmed up")
    
    results = []

    for n_atoms, indices in tqdm(indices_by_natoms.items(), desc="Processing N_atoms"):
        if len(indices) == 0:
            continue
        n_atoms = int(n_atoms)

        # Limit number of samples
        indices_to_test = indices[:max_samples_per_n]

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)

        for _batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            batch = _batch.clone().to(device)
            batch = compute_extra_props(batch)

            # Time prediction
            time_prediction, mem_prediction = time_hessian_computation(
                model, batch, "prediction"
            )
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "prediction",
                    "time": time_prediction,
                    "memory": mem_prediction,
                }
            )
            
            # clear memory
            torch.cuda.empty_cache()
            
            # fresh batch
            batch = _batch.clone().to(device)
            batch = compute_extra_props(batch)
            
            # Time autograd
            time_autograd, mem_autograd = time_hessian_computation(
                model, batch, "autograd"
            )
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "autograd",
                    "time": time_autograd,
                    "memory": mem_autograd,
                }
            )
            
            # clear memory
            torch.cuda.empty_cache()

    # Save results
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_speed_comparison_results.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return results_df


def plot_speed_comparison(results_df, output_dir="./resultseval"):
    output_dir = Path(output_dir)
    # Plot results for speed
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    std_times = results_df.groupby(["n_atoms", "method"])["time"].std().unstack()

    fig = go.Figure()
    for method in avg_times.columns:
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method],
                mode="lines+markers",
                name=method,
                # error_y=dict(type="data", array=std_times[method]),
            )
        )

    fig.update_layout(
        title="Hessian Computation Speed: Autograd vs. Prediction",
        xaxis_title="Number of Atoms (N)",
        yaxis_title="Average Time (ms)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40)
    )
    output_path = output_dir / "speed_comparison_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
    output_path = output_dir / "speed_comparison_plot.png"
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")

    # Plot results for memory
    plot_memory_usage(results_df, output_dir)


def plot_memory_usage(results_df, output_dir="./resultseval"):
    output_dir = Path(output_dir)
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()

    fig = go.Figure()
    for method in avg_memory.columns:
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory[method],
                mode="lines+markers",
                name=method,
            )
        )

    fig.update_layout(
        title="Hessian Computation Memory Usage: Autograd vs. Prediction",
        xaxis_title="Number of Atoms (N)",
        yaxis_title="Peak Memory (MB)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40)
    )
    output_path = output_dir / "memory_usage_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
    output_path = output_dir / "memory_usage_plot.png"
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")


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
