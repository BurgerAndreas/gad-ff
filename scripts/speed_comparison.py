import torch
from torch_geometric.loader import DataLoader as TGDataLoader
from torch.utils.data import Subset
import argparse
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go


import json
from pathlib import Path
import os
from collections import defaultdict

from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path
from ocpmodels.hessian_graph_transform import HessianGraphTransform, FOLLOW_BATCH
from gadff.horm.training_module import SchemaUniformDataset
from gadff.colours import HESSIAN_METHOD_TO_COLOUR


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
    do_autograd = hessian_method == "autograd"
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if "equiformer" in model.name.lower():
        if do_autograd:
            batch.pos.requires_grad_()
            ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
            compute_hessian(batch.pos, ener, force)
        else:
            with torch.no_grad():
                # for a fair comparison
                # compute graph and Hessian indices on the fly
                ener, force, out = model.forward(
                    batch, otf_graph=True, hessian=True, add_props=True
                )
    else:
        batch.pos.requires_grad_()
        ener, force, out = model.forward(batch)
        compute_hessian(batch.pos, ener, force)

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
    output_dir="./results_speed",
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
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset = LmdbDataset(fix_dataset_path(dataset_name), transform=transform)
    dataset = SchemaUniformDataset(dataset)

    # do a couple of forward passes to warm up the model
    # populate caches, jit, load cuda kernels, and what not
    loader = TGDataLoader(
        dataset, batch_size=1, shuffle=False, follow_batch=FOLLOW_BATCH
    )
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
        loader = TGDataLoader(
            subset, batch_size=1, shuffle=False, follow_batch=FOLLOW_BATCH
        )

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


def plot_speed_comparison(results_df, output_dir="./results_speed"):
    output_dir = Path(output_dir)
    # Plot results for speed
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()

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
        margin=dict(l=40, r=40, b=40, t=40),
    )
    # Add arrow from the largest autograd value to the largest prediction value
    if "autograd" in avg_times.columns and "prediction" in avg_times.columns:
        try:
            _auto = avg_times["autograd"].dropna()
            _pred = avg_times["prediction"].dropna()
            if len(_auto) > 0 and len(_pred) > 0:
                _x_auto = _auto.idxmax()
                _y_auto = _auto.loc[_x_auto]
                _x_pred = _pred.idxmax()
                _y_pred = _pred.loc[_x_pred]
                fig.add_annotation(
                    x=_x_pred,
                    y=_y_pred,
                    ax=_x_auto,
                    ay=_y_auto,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="rgba(50,50,50,0.6)",
                )
        except Exception:
            pass
    output_path = output_dir / "speed_comparison_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
    output_path = output_dir / "speed_comparison_plot.png"
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")

    # Plot results for memory
    plot_memory_usage(results_df, output_dir)


def plot_memory_usage(results_df, output_dir="./results_speed"):
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
        margin=dict(l=40, r=40, b=40, t=40),
    )
    # Add arrow from the largest autograd value to the largest prediction value
    if "autograd" in avg_memory.columns and "prediction" in avg_memory.columns:
        try:
            _auto = avg_memory["autograd"].dropna()
            _pred = avg_memory["prediction"].dropna()
            if len(_auto) > 0 and len(_pred) > 0:
                _x_auto = _auto.idxmax()
                _y_auto = _auto.loc[_x_auto]
                _x_pred = _pred.idxmax()
                _y_pred = _pred.loc[_x_pred]
                fig.add_annotation(
                    x=_x_pred,
                    y=_y_pred,
                    ax=_x_auto,
                    ay=_y_auto,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="rgba(50,50,50,0.8)",
                )
        except Exception:
            pass
    output_path = output_dir / "memory_usage_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
    output_path = output_dir / "memory_usage_plot.png"
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")


# ---------------------------------
# Prediction vs Batch Size Benchmark
# ---------------------------------
def prediction_batchsize_benchmark(
    checkpoint_path,
    dataset_name,
    # bz 128 only sometimes fits into memory of a RTX3060
    batch_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64),
    num_batches=10,
    device="cuda",
    output_path="./results_speed/speed_bz.csv",
    max_autograd_batch_size=8,
):
    """Benchmark Hessian prediction speed vs batch size using random batches of any N atoms.

    Returns a DataFrame with columns: batch_size, time, memory
    """
    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name

    # Prepare dataset (same transform settings as speed_comparison)
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset = LmdbDataset(fix_dataset_path(dataset_name), transform=transform)
    dataset = SchemaUniformDataset(dataset)

    # Light warm-up
    warm_loader = TGDataLoader(
        dataset, batch_size=1, shuffle=True, follow_batch=FOLLOW_BATCH
    )
    for i, sample in enumerate(warm_loader):
        batch = sample.to(device)
        batch = compute_extra_props(batch)
        time_hessian_computation(model, batch, "prediction")
        torch.cuda.empty_cache()
        time_hessian_computation(model, batch, "autograd")
        torch.cuda.empty_cache()
        if i >= 5:
            break
    del warm_loader
    # gc.collect()
    # torch.cuda.empty_cache()

    results = []
    dataset_len = len(dataset)
    for bsz in batch_sizes:
        print(f"Batch size: {bsz}")
        # Prepare a subset with random indices; allow duplicates via randint
        num_needed = num_batches * bsz
        if dataset_len == 0:
            break
        rand_idx = torch.randint(
            low=0, high=dataset_len, size=(num_needed,), dtype=torch.long
        ).tolist()
        subset = Subset(dataset, rand_idx)
        loader = TGDataLoader(
            subset, batch_size=bsz, shuffle=True, follow_batch=FOLLOW_BATCH
        )

        torch.cuda.empty_cache()

        measured = 0
        for sample in loader:
            batch = sample.clone().to(device)
            batch = compute_extra_props(batch)

            # Time prediction
            time_prediction, mem_prediction = time_hessian_computation(
                model, batch, "prediction"
            )
            results.append(
                {
                    # "n_atoms": n_atoms,
                    "method": "prediction",
                    "time": time_prediction,
                    "memory": mem_prediction,
                    "batch_size": bsz,
                }
            )

            # clear memory
            torch.cuda.empty_cache()

            if bsz < max_autograd_batch_size:
                # fresh batch
                batch = sample.clone().to(device)
                batch = compute_extra_props(batch)

                # Time autograd
                time_autograd, mem_autograd = time_hessian_computation(
                    model, batch, "autograd"
                )
                results.append(
                    {
                        # "n_atoms": n_atoms,
                        "method": "autograd",
                        "time": time_autograd,
                        "memory": mem_autograd,
                        "batch_size": bsz,
                    }
                )

                # clear memory
                torch.cuda.empty_cache()

            msg = f"Batch size={bsz}, avg n_atoms={batch.n_atoms.mean():.1f}"
            msg += f", prediction={time_prediction:.3f} ms"
            if bsz < max_autograd_batch_size:
                msg += f", autograd={time_autograd:.3f} ms"
            print(msg)

            measured += 1
            if measured >= num_batches:
                break

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Batch size benchmark results saved to {output_path}")
    return results_df


def plot_prediction_batchsize(results_df, output_dir="./results_speed", logy=False):
    output_dir = Path(output_dir)
    width = 1600
    height = 450
    fig = go.Figure()
    # prediction
    pred_results_df = results_df[results_df["method"] == "prediction"]
    avg_times = pred_results_df.groupby(["batch_size"])["time"].mean()
    fig.add_trace(
        go.Scatter(
            x=avg_times.index,
            y=avg_times.values,
            mode="lines+markers",
            name="prediction",
            # error_y=dict(type="data", array=std_times.values),
        )
    )
    # autograd
    autograd_results_df = results_df[results_df["method"] == "autograd"]
    avg_times_autograd = autograd_results_df.groupby(["batch_size"])["time"].mean()
    fig.add_trace(
        go.Scatter(
            x=avg_times_autograd.index,
            y=avg_times_autograd.values,
            mode="lines+markers",
            name="autograd",
        )
    )
    fig.update_layout(
        title="Prediction Speed vs Batch Size",
        xaxis_title="Batch Size",
        yaxis_title="Average Time (ms)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    # output_path = output_dir / "prediction_batchsize_plot.html"
    # fig.write_html(output_path)
    # print(f"Plot saved to {output_path}")
    output_path = output_dir / "prediction_batchsize_plot.png"
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to {output_path}")

    #######################
    # plot time per sample
    fig = go.Figure()
    # prediction
    avg_times = (
        pred_results_df.groupby(["batch_size"])["time"].mean()
        / pred_results_df.groupby(["batch_size"])["batch_size"].mean()
    )
    fig.add_trace(
        go.Scatter(
            x=avg_times.index,
            y=avg_times.values,
            mode="lines+markers",
            name="prediction",
            # error_y=dict(type="data", array=std_times.values),
        )
    )
    # autograd
    avg_times_autograd = (
        autograd_results_df.groupby(["batch_size"])["time"].mean()
        / autograd_results_df.groupby(["batch_size"])["batch_size"].mean()
    )
    fig.add_trace(
        go.Scatter(
            x=avg_times_autograd.index,
            y=avg_times_autograd.values,
            mode="lines+markers",
            name="autograd",
        )
    )
    fig.update_layout(
        # title="Hessian Prediction Speed vs Batch Size",
        xaxis_title="Batch Size",
        yaxis_title="Average Time per Sample (ms)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
        width=width,
        height=height,
        legend=dict(
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)",
        ),
        yaxis=dict(type="log") if logy else None,
    )
    # output_path = output_dir / "prediction_batchsize_plot.html"
    # fig.write_html(output_path)
    # print(f"Plot saved to {output_path}")
    output_path = (
        output_dir
        / f"prediction_batchsize_time_per_sample{'_logy' if logy else ''}.png"
    )
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")


def plot_combined_speed_memory_batchsize(
    results_df, bz_results_df, output_dir="./results_speed"
):
    from plotly.subplots import make_subplots

    output_dir = Path(output_dir)
    height = 400
    width = height * 3

    # Map method names to colours (handle both "predict" and "prediction")
    def _color_for_method(method):
        key = method
        if method == "prediction":
            key = "predict"
        return HESSIAN_METHOD_TO_COLOUR.get(key)

    # Aggregations for speed and memory vs N
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()

    # Aggregation for prediction vs batch size (per-sample)
    autograd_results_df = bz_results_df[bz_results_df["method"] == "autograd"]
    pred_results_df = bz_results_df[bz_results_df["method"] == "prediction"]
    pred_avg_times = (
        pred_results_df.groupby(["batch_size"])["time"].mean()
        / pred_results_df.groupby(["batch_size"])["batch_size"].mean()
    )
    autograd_avg_times = (
        autograd_results_df.groupby(["batch_size"])["time"].mean()
        / autograd_results_df.groupby(["batch_size"])["batch_size"].mean()
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        # subplot_titles=("", "", ""),
        horizontal_spacing=0.03,
        vertical_spacing=0.0,
    )

    # # Add subplot labels
    # labels = ['d', 'e', 'f']
    # for ax, label in zip(axes, labels):
    #     ax.text(-0.1, 1.05, label, transform=ax.transAxes,
    #             fontsize=14, fontweight='bold', va='top', ha='right')

    # Col 1: Speed vs N
    for method in avg_times.columns:
        color = _color_for_method(method)
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method] / 1000.0,
                mode="lines+markers",
                name=method,
                legend="legend",
                showlegend=True,
                line=dict(color=color) if color else None,
                marker=dict(color=color) if color else None,
            ),
            row=1,
            col=1,
        )

    # Col 2: Memory vs N
    for method in avg_memory.columns:
        color = _color_for_method(method)
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory[method],
                mode="lines+markers",
                name=method,
                legend="legend2",
                showlegend=True,
                line=dict(color=color) if color else None,
                marker=dict(color=color) if color else None,
            ),
            row=1,
            col=2,
        )

    # Add arrows for subplots 1 and 2: from max autograd to max prediction
    if "autograd" in avg_times.columns and "prediction" in avg_times.columns:
        _auto = avg_times["autograd"].dropna()
        _pred = avg_times["prediction"].dropna()
        if len(_auto) > 0 and len(_pred) > 0:
            _x_auto = _auto.idxmax()
            _y_auto = _auto.loc[_x_auto]
            _x_pred = _pred.idxmax()
            _y_pred = _pred.loc[_x_pred]
            # Shorten arrow a bit to avoid overlapping the destination
            _x_head = _x_auto + (_x_pred - _x_auto) * 0.97
            # Convert to seconds for plotting
            _y_auto_s = _y_auto / 1000.0
            _y_pred_s = _y_pred / 1000.0
            _y_head_s = _y_auto_s + (_y_pred_s - _y_auto_s) * 0.97
            fig.add_annotation(
                x=_x_head,
                y=_y_head_s,
                ax=_x_auto,
                ay=_y_auto_s,
                xref="x1",
                yref="y1",
                axref="x1",
                ayref="y1",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(50,50,50,0.8)",
            )

    if "autograd" in avg_memory.columns and "prediction" in avg_memory.columns:
        _auto_m = avg_memory["autograd"].dropna()
        _pred_m = avg_memory["prediction"].dropna()
        if len(_auto_m) > 0 and len(_pred_m) > 0:
            _x_auto_m = _auto_m.idxmax()
            _y_auto_m = _auto_m.loc[_x_auto_m]
            _x_pred_m = _pred_m.idxmax()
            _y_pred_m = _pred_m.loc[_x_pred_m]
            _x_head_m = _x_auto_m + (_x_pred_m - _x_auto_m) * 0.97
            _y_head_m = _y_auto_m + (_y_pred_m - _y_auto_m) * 0.97
            fig.add_annotation(
                x=_x_head_m,
                y=_y_head_m,
                ax=_x_auto_m,
                ay=_y_auto_m,
                xref="x2",
                yref="y2",
                axref="x2",
                ayref="y2",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(50,50,50,0.8)",
            )

    # Col 3: Prediction vs Batch Size (per-sample)
    # Ensure strictly positive values for log scale
    pred_avg_times_plot = pred_avg_times.copy()
    autograd_avg_times_plot = autograd_avg_times.copy()

    color = _color_for_method("autograd")
    fig.add_trace(
        go.Scatter(
            x=autograd_avg_times_plot.index,
            y=autograd_avg_times_plot.values,
            mode="lines+markers",
            name="autograd",
            legend="legend3",
            showlegend=True,
            line=dict(color=color) if color else None,
            marker=dict(color=color) if color else None,
        ),
        row=1,
        col=3,
    )
    color = _color_for_method("prediction")
    fig.add_trace(
        go.Scatter(
            x=pred_avg_times_plot.index,
            y=pred_avg_times_plot.values,
            mode="lines+markers",
            name="prediction",
            legend="legend3",
            showlegend=True,
            line=dict(color=color) if color else None,
            marker=dict(color=color) if color else None,
        ),
        row=1,
        col=3,
    )

    # derive subplot domains to place legends at the top-middle of each subplot
    dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.3]
    dom2 = fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.35, 0.65]
    dom3 = fig.layout.xaxis3.domain if hasattr(fig.layout, "xaxis3") else [0.7, 1.0]
    x1 = 0.5 * (dom1[0] + dom1[1])
    x2 = 0.5 * (dom2[0] + dom2[1])
    x3 = 0.5 * (dom3[0] + dom3[1])

    # Add axis titles for each subplot
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=1, row=1, col=1)
    fig.update_yaxes(title_text="Average Time (s)", title_standoff=10, row=1, col=1)
    # fig.update_yaxes(tickformat=".0e", exponentformat="e",  row=1, col=1)
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=1, row=1, col=2)
    fig.update_yaxes(title_text="Peak Memory (MB)", title_standoff=0, row=1, col=2)
    fig.update_xaxes(title_text="Batch Size", title_standoff=1, row=1, col=3)
    fig.update_yaxes(
        title_text="Average Time per Sample (ms)", title_standoff=0, row=1, col=3
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=0, b=0, t=20),
        # margin=dict(l=0, r=0, b=0, t=0),
        width=width,
        height=height,
        yaxis3=dict(type="log"),
        legend=dict(
            x=x1,
            y=0.98,
            xanchor="center",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)",
        ),
        legend2=dict(
            x=x2,
            y=0.98,
            xanchor="center",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)",
        ),
        legend3=dict(
            x=x3,
            y=0.98,
            xanchor="center",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)",
        ),
    )

    # Add subplot panel labels (a, b, c) at top-left outside each subplot
    fig.add_annotation(
        x=dom1[0],  # -0.005
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>a</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=14),
    )
    fig.add_annotation(
        x=dom2[0],
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>b</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=14),
    )
    fig.add_annotation(
        x=dom3[0],
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>c</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=14),
    )

    # add manual arrow for subplot 3 in normalized domain coords
    fig.add_annotation(
        x=0.93,  # lower means left
        y=0.1,  # lower means lower
        # tail
        ax=0.08,
        ay=0.71,  # lower means lower
        xref="x3 domain",
        yref="y3 domain",
        # what coordinates the tail of the annotation (ax,ay) is specified
        axref="x3 domain",
        ayref="y3 domain",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="rgba(50,50,50,0.8)",
    )

    # Save only PNG to keep output concise
    output_path = output_dir / "combined_speed_memory_batchsize.png"
    # The height of the exported image in layout pixels. If the scale property is 1.0, this will also be the height of the exported image in physical pixels.
    # Scale > 1 increases the image resolution
    fig.write_image(output_path, width=width, height=height)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    """
    python scripts/speed_comparison.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt
    python scripts/speed_comparison.py --dataset ts1x-val.lmdb --max_samples_per_n 100
    python scripts/speed_comparison.py --dataset ts1x_hess_train_big.lmdb --max_samples_per_n 1000
    """
    parser = argparse.ArgumentParser(description="Speed comparison")

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
        "--max_autograd_bz",
        type=int,
        default=8,
        help="Maximum batch size for autograd.",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the speed comparison. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--redobz",
        type=bool,
        default=False,
        help="Redo the speed comparison. If false attempt to load existing results.",
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    redo = args.redo

    output_dir = "./results_speed"
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

    ##############################################################
    # Second benchmark: prediction-only vs batch size (random N)
    print()

    # dataset_name = args.dataset
    dataset_name = "ts1x-val.lmdb"
    output_dir = Path("./results_speed")
    output_path_speedbz = (
        output_dir
        / f"{dataset_name}_prediction_batchsize_results_agbz{args.max_autograd_bz}.csv"
    )
    if output_path_speedbz.exists() and not args.redobz:
        bz_results_df = pd.read_csv(output_path_speedbz)
        print(
            f"Loaded existing prediction batch-size results from {output_path_speedbz}"
        )
    else:
        bz_results_df = prediction_batchsize_benchmark(
            checkpoint_path=args.ckpt_path,
            dataset_name=dataset_name,
            output_path=output_path_speedbz,
            max_autograd_batch_size=args.max_autograd_bz,
        )

    plot_prediction_batchsize(bz_results_df, output_dir=output_dir)
    plot_prediction_batchsize(bz_results_df, output_dir=output_dir, logy=True)

    # Combined side-by-side plot
    plot_combined_speed_memory_batchsize(
        results_df, bz_results_df, output_dir=output_dir
    )

    print("Done.")
