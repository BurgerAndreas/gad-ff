"""
Evaluate models on DFT geometries (from compute_dft_geometries.py) to measure
accuracy vs molecular size. Saves per-sample metrics CSV for downstream analysis
(e.g., error distribution over natoms).

Usage:
    uv run scripts/size_eval.py -c ckpt/eqv2.ckpt
    uv run scripts/size_eval.py -c ckpt/eqv2.ckpt -d geometries/dft_geometries.lmdb -hm predict
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as TGDataLoader
from tqdm import tqdm

from alphanet.models.alphanet import AlphaNet
from leftnet.model.leftnet import LEFTNet

from gadff.horm.training_module import PotentialModule
from gadff.horm.ff_lmdb import LmdbDataset
from nets.prediction_utils import compute_extra_props, Z_TO_ATOM_SYMBOL
from ocpmodels.hessian_graph_transform import HessianGraphTransform

from gadff.frequency_analysis import analyze_frequencies, eigval_to_wavenumber
from pathlib import Path


def find_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    checkpoint_dir = Path("checkpoint/hip")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint path {checkpoint_path} does not exist and "
            f"checkpoint directory {checkpoint_dir} not found"
        )

    matching_dirs = [
        d for d in checkpoint_dir.iterdir() if d.is_dir() and checkpoint_path in d.name
    ]

    if not matching_dirs:
        raise FileNotFoundError(
            f"Checkpoint path {checkpoint_path} does not exist and "
            f"no matching directory found in {checkpoint_dir} containing '{checkpoint_path}'"
        )

    if len(matching_dirs) > 1:
        print(
            f"Warning: Multiple matching directories found: {[d.name for d in matching_dirs]}"
        )
        print(f"Using: {matching_dirs[0].name}")

    found_ckpt = matching_dirs[0] / "last.ckpt"
    if not found_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint path {checkpoint_path} does not exist and "
            f"last.ckpt not found in {matching_dirs[0]}"
        )

    print(f"Found checkpoint: {found_ckpt}")
    return str(found_ckpt)


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)

    n_comp = forces.reshape(-1).shape[0]
    hess = []
    for f in forces.reshape(-1):
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)

    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def evaluate(
    data_path,
    checkpoint_path,
    hessian_method,
    max_samples=None,
    redo=False,
):
    checkpoint_path = find_checkpoint(checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model_config = ckpt["hyper_parameters"]["model_config"]
    print(f"Model name: {model_name}")

    # Create results file path
    dataset_name = os.path.basename(data_path).split(".")[0]
    results_dir = "results_size_eval"
    os.makedirs(results_dir, exist_ok=True)
    ckpt_name = os.path.basename(checkpoint_path).split(".")[0]
    results_file = (
        f"{results_dir}/{ckpt_name}_{dataset_name}_{hessian_method}_metrics.csv"
    )

    if os.path.exists(results_file) and not redo:
        print(f"Loading existing results from {results_file}")
        df_results = pd.read_csv(results_file)
    else:
        torch.manual_seed(42)
        np.random.seed(42)

        model = PotentialModule.load_from_checkpoint(
            checkpoint_path,
            strict=False,
        ).potential.to("cuda")
        model.eval()

        do_autograd = hessian_method == "autograd"
        print(f"do_autograd: {do_autograd}")

        if hessian_method == "predict":
            transform = HessianGraphTransform(
                cutoff=model.cutoff,
                cutoff_hessian=model.cutoff_hessian,
                max_neighbors=model.max_neighbors,
                use_pbc=model.use_pbc,
            )
        else:
            transform = None

        # Load LMDB directly (no fix_dataset_path since path is given directly)
        dataset = LmdbDataset(data_path, transform=transform)
        dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

        n_total_samples = len(dataloader)
        if max_samples is not None:
            n_total_samples = min(max_samples, n_total_samples)

        # Warmup
        for _i, batch in tqdm(enumerate(dataloader), desc="Warmup", total=10):
            if _i >= 10:
                break
            batch = batch.to("cuda")
            batch = compute_extra_props(batch)
            n_atoms = batch.pos.shape[0]

            if model_name == "LEFTNet":
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward_autograd(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)
            elif "equiformer" in model_name.lower():
                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=False
                    )
                    hessian_model = compute_hessian(
                        batch.pos, energy_model, force_model
                    )
                else:
                    with torch.no_grad():
                        energy_model, force_model, out = model.forward(
                            batch, otf_graph=False, hessian=True, add_props=True
                        )
                    hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)
            else:
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)

        sample_metrics = []
        n_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating", total=n_total_samples):
            batch = batch.to("cuda")
            batch = compute_extra_props(batch)

            n_atoms = batch.pos.shape[0]

            sample_data = {
                "sample_idx": n_samples,
                "natoms": n_atoms,
            }

            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # Forward pass
            if model_name == "LEFTNet":
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward_autograd(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)
            elif "equiformer" in model_name.lower():
                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=False
                    )
                    hessian_model = compute_hessian(
                        batch.pos, energy_model, force_model
                    )
                else:
                    with torch.no_grad():
                        energy_model, force_model, out = model.forward(
                            batch, otf_graph=False, hessian=True, add_props=True
                        )
                    hessian_model = out["hessian"]
            else:
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)

            end_event.record()
            torch.cuda.synchronize()

            time_taken = start_event.elapsed_time(end_event)
            memory_usage = torch.cuda.max_memory_allocated() / 1e6
            sample_data["time"] = time_taken
            sample_data["memory"] = memory_usage

            hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)

            # Eigenspectra
            eigvals_model, eigvecs_model = torch.linalg.eigh(hessian_model)

            # Energy error
            energy_true = batch.energy
            e_mae = torch.mean(
                torch.abs(energy_model.squeeze() - energy_true.squeeze())
            )
            e_mae_per_atom = e_mae / n_atoms
            sample_data["energy_mae"] = e_mae.item()
            sample_data["energy_mae_per_atom"] = e_mae_per_atom.item()

            # Force error
            f_mae = torch.mean(torch.abs(force_model - batch.forces))
            sample_data["forces_mae"] = f_mae.item()

            # Hessian error
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
            h_mae = torch.mean(torch.abs(hessian_model - hessian_true))
            sample_data["hessian_mae"] = h_mae.item()
            h_mre = torch.mean(
                torch.abs(hessian_model - hessian_true)
                / (torch.abs(hessian_true) + 1e-8)
            )
            sample_data["hessian_mre"] = h_mre.item()

            # Eigenvalue error
            eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)
            eigval_mae = torch.mean(torch.abs(eigvals_model - eigvals_true))
            sample_data["eigval_mae"] = eigval_mae.item()
            eigval_mre = torch.mean(
                torch.abs(eigvals_model - eigvals_true)
                / (torch.abs(eigvals_true) + 1e-8)
            )
            sample_data["eigval_mre"] = eigval_mre.item()

            # Asymmetry
            asymmetry_mae = torch.mean(torch.abs(hessian_model - hessian_model.T))
            true_asymmetry_mae = torch.mean(torch.abs(hessian_true - hessian_true.T))
            sample_data["asymmetry_mae"] = asymmetry_mae.item()
            sample_data["true_asymmetry_mae"] = true_asymmetry_mae.item()

            # Eckart-projected frequency analysis
            true_freqs = analyze_frequencies(
                hessian=hessian_true.detach().cpu().numpy(),
                cart_coords=batch.pos.detach().cpu().numpy(),
                atomsymbols=[Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z],
            )
            true_eigvecs_eckart = torch.tensor(true_freqs["eigvecs"])
            true_eigvals_eckart = torch.tensor(true_freqs["eigvals"])

            freqs_model = analyze_frequencies(
                hessian=hessian_model.detach().cpu().numpy(),
                cart_coords=batch.pos.detach().cpu().numpy(),
                atomsymbols=[Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z],
            )
            eigvecs_model_eckart = torch.tensor(freqs_model["eigvecs"])
            eigvals_model_eckart = torch.tensor(freqs_model["eigvals"])

            sample_data["true_neg_num"] = true_freqs["neg_num"]
            sample_data["true_is_minima"] = 1 if true_freqs["neg_num"] == 0 else 0
            sample_data["true_is_ts"] = 1 if true_freqs["neg_num"] == 1 else 0
            sample_data["true_is_ts_order2"] = 1 if true_freqs["neg_num"] == 2 else 0
            sample_data["true_is_higher_order"] = 1 if true_freqs["neg_num"] > 2 else 0
            sample_data["model_neg_num"] = freqs_model["neg_num"]
            sample_data["model_is_minima"] = 1 if freqs_model["neg_num"] == 0 else 0
            sample_data["model_is_ts"] = 1 if freqs_model["neg_num"] == 1 else 0
            sample_data["model_is_ts_order2"] = 1 if freqs_model["neg_num"] == 2 else 0
            sample_data["model_is_higher_order"] = 1 if freqs_model["neg_num"] > 2 else 0
            sample_data["neg_num_agree"] = (
                1 if true_freqs["neg_num"] == freqs_model["neg_num"] else 0
            )

            sample_data["eigval_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart - true_eigvals_eckart)
            ).item()
            sample_data["eigval_mre_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart - true_eigvals_eckart)
                / (torch.abs(true_eigvals_eckart) + 1e-8)
            ).item()
            sample_data["eigval1_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[0] - true_eigvals_eckart[0])
            ).item()
            sample_data["eigval1_mre_eckart"] = (
                torch.abs(eigvals_model_eckart[0] - true_eigvals_eckart[0])
                / (torch.abs(true_eigvals_eckart[0]) + 1e-8)
            ).item()
            sample_data["eigval2_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[1] - true_eigvals_eckart[1])
            ).item()
            sample_data["eigval2_mre_eckart"] = (
                torch.abs(eigvals_model_eckart[1] - true_eigvals_eckart[1])
                / (torch.abs(true_eigvals_eckart[1]) + 1e-8)
            ).item()
            sample_data["eigvec1_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 0], true_eigvecs_eckart[:, 0])
            ).item()
            sample_data["eigvec2_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 1], true_eigvecs_eckart[:, 1])
            ).item()

            # Global eigenvector overlap: ||abs(Q_model @ Q_true^T) - I||_F
            M = eigvecs_model_eckart.T @ true_eigvecs_eckart
            sample_data["eigvec_overlap_error"] = torch.norm(
                M.abs() - torch.eye(M.shape[0]), p="fro"
            ).item()

            # Vibrational frequency MAE (400-4000 cm⁻¹)
            true_eigvals_np = true_eigvals_eckart.detach().cpu().numpy()
            model_eigvals_np = eigvals_model_eckart.detach().cpu().numpy()
            true_wavenumbers = eigval_to_wavenumber(true_eigvals_np)
            model_wavenumbers = eigval_to_wavenumber(model_eigvals_np)

            true_mask = (
                (true_wavenumbers >= 400)
                & (true_wavenumbers <= 4000)
                & (true_eigvals_np > 0)
            )
            model_mask = (
                (model_wavenumbers >= 400)
                & (model_wavenumbers <= 4000)
                & (model_eigvals_np > 0)
            )
            combined_mask = true_mask & model_mask

            if combined_mask.sum() > 0:
                sample_data["freq_mae_400_4000"] = np.mean(
                    np.abs(
                        model_wavenumbers[combined_mask]
                        - true_wavenumbers[combined_mask]
                    )
                )
            else:
                sample_data["freq_mae_400_4000"] = np.nan

            sample_metrics.append(sample_data)
            n_samples += 1

            torch.cuda.empty_cache()

            if max_samples is not None and n_samples >= max_samples:
                break

        df_results = pd.DataFrame(sample_metrics)
        df_results.to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")

    # Print aggregated summary
    print("\n=== Aggregated Results ===")
    for col in df_results.columns:
        if pd.api.types.is_numeric_dtype(df_results[col]):
            print(f"  {col}: {df_results[col].mean():.4f}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on DFT geometries (size scaling)"
    )
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
        default="geometries/dft_geometries.lmdb",
        help="Path to DFT geometries LMDB",
    )
    parser.add_argument(
        "--hessian_method",
        "-hm",
        type=str,
        default="autograd",
        help="Hessian computation method (autograd or predict)",
    )
    parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--redo",
        "-r",
        action="store_true",
        help="Re-run eval even if results CSV exists",
    )
    args = parser.parse_args()

    df_results = evaluate(
        data_path=args.dataset,
        checkpoint_path=args.ckpt_path,
        hessian_method=args.hessian_method,
        max_samples=args.max_samples,
        redo=args.redo,
    )
