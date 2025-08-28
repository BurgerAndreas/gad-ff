import torch
from tqdm import tqdm
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch_geometric.loader import DataLoader as TGDataLoader

from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path
from ocpmodels.hessian_graph_transform import HessianGraphTransform

from ReactBench.utils.frequency_analysis import analyze_frequencies

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


def evaluate(
    lmdb_path,
    checkpoint_path,
    config_path, # not used
    hessian_method,
    max_samples=None,
    wandb_run_id=None,
    wandb_kwargs={},
    redo=False,
):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model_config = ckpt["hyper_parameters"]["model_config"]
    print(f"Model name: {model_name}")

    _name = "hormeval"
    _name += "_" + checkpoint_path.split("/")[-2]
    _name += "_" + checkpoint_path.split("/")[-1].split(".")[0]
    _name += "_" + lmdb_path.split("/")[-1].split(".")[0]
    _name += "_" + hessian_method

    if wandb_run_id is None:
        wandb.init(
            project="horm",
            name=_name,
            config={
                "checkpoint": checkpoint_path,
                "dataset": lmdb_path,
                "max_samples": max_samples,
                "model_name": model_name,
                "config_path": config_path,
                "hessian_method": hessian_method,
                "model_config": model_config,
            },
            tags=["hormmetrics"],
            **wandb_kwargs,
        )

    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to("cuda")
    model.eval()

    do_autograd = hessian_method == "autograd"
    print(f"do_autograd: {do_autograd}")

    # Create results file path
    dataset_name = lmdb_path.split("/")[-1].split(".")[0]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    ckpt_name = checkpoint_path.split("/")[-1].split(".")[0]
    results_file = (
        f"{results_dir}/{ckpt_name}_{dataset_name}_{hessian_method}_metrics.csv"
    )

    # Check if results already exist and redo is False
    if os.path.exists(results_file) and not redo:
        print(f"Loading existing results from {results_file}")
        df_results = pd.read_csv(results_file)

    else:
        # if hessian_method == "predict" or model.do_hessian or model.otf_graph == False:
        if hessian_method == "predict":
            transform = HessianGraphTransform(
                cutoff=model.cutoff,
                cutoff_hessian=model.cutoff_hessian,
                max_neighbors=model.max_neighbors,
                use_pbc=model.use_pbc,
            )
        else:
            transform = None

        dataset = LmdbDataset(fix_dataset_path(lmdb_path), transform=transform)
        dataloader = TGDataLoader(dataset, batch_size=1, shuffle=True)

        # Initialize metrics collection for per-sample DataFrame
        sample_metrics = []
        n_samples = 0

        if max_samples is not None:
            total = max_samples
        else:
            total = len(dataloader)

        for batch in tqdm(dataloader, desc="Evaluating", total=total):
            batch = batch.to("cuda")
            batch = compute_extra_props(batch)

            n_atoms = batch.pos.shape[0]

            # Forward pass
            if model_name == "LEFTNet":
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward_autograd(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)
            elif "equiformer" in model_name.lower():
                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=True, hessian=False
                    )
                    hessian_model = compute_hessian(
                        batch.pos, energy_model, force_model
                    )
                else:
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=True
                    )
                    hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)
            else:
                batch.pos.requires_grad_()
                energy_model, force_model, out = model.forward(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)

            # Compute hessian eigenspectra
            eigvals_model, eigvecs_model = torch.linalg.eigh(hessian_model)

            # Compute errors
            e_error = torch.mean(torch.abs(energy_model.squeeze() - batch.ae))
            f_error = torch.mean(torch.abs(force_model - batch.forces))

            # Reshape true hessian
            n_atoms = batch.pos.shape[0]
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
            h_error = torch.mean(torch.abs(hessian_model - hessian_true))

            # Eigenvalue error
            eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)

            # Asymmetry error
            asymmetry_error = torch.mean(torch.abs(hessian_model - hessian_model.T))
            true_asymmetry_error = torch.mean(torch.abs(hessian_true - hessian_true.T))

            # Additional metrics
            eigval_mae = torch.mean(
                torch.abs(eigvals_model - eigvals_true)
            )  # eV/Angstrom^2
            eigval1_mae = torch.mean(torch.abs(eigvals_model[0] - eigvals_true[0]))
            eigval2_mae = torch.mean(torch.abs(eigvals_model[1] - eigvals_true[1]))
            eigvec1_mae = torch.mean(
                torch.abs(eigvecs_model[:, 0] - eigvecs_true[:, 0])
            )
            eigvec2_mae = torch.mean(
                torch.abs(eigvecs_model[:, 1] - eigvecs_true[:, 1])
            )
            eigvec1_cos = torch.abs(torch.dot(eigvecs_model[:, 0], eigvecs_true[:, 0]))
            eigvec2_cos = torch.abs(torch.dot(eigvecs_model[:, 1], eigvecs_true[:, 1]))

            # Collect per-sample metrics
            sample_data = {
                "sample_idx": n_samples,
                "natoms": n_atoms,
                "energy_error": e_error.item(),
                "forces_error": f_error.item(),
                "hessian_error": h_error.item(),
                "asymmetry_error": asymmetry_error.item(),
                "true_asymmetry_error": true_asymmetry_error.item(),
                "eigval_mae": eigval_mae.item(),
                "eigval1_mae": eigval1_mae.item(),
                "eigval2_mae": eigval2_mae.item(),
                "eigvec1_mae": eigvec1_mae.item(),
                "eigvec2_mae": eigvec2_mae.item(),
                "eigvec1_cos": eigvec1_cos.item(),
                "eigvec2_cos": eigvec2_cos.item(),
            }

            true_freqs = analyze_frequencies(
                hessian=hessian_true,
                cart_coords=batch.pos,
                atomsymbols=batch.atom_types,
            )
            true_neg_num = true_freqs["neg_num"]

            freqs_model = analyze_frequencies(
                hessian=hessian_model,
                cart_coords=batch.pos,
                atomsymbols=batch.atom_types,
            )
            freqs_model_neg_num = freqs_model["neg_num"]

            sample_data["true_neg_num"] = true_neg_num
            sample_data["true_is_ts"] = 1 if true_neg_num == 1 else 0
            sample_data["model_neg_num"] = freqs_model_neg_num
            sample_data["model_is_ts"] = 1 if freqs_model_neg_num == 1 else 0
            sample_data["neg_num_agree"] = (
                1 if (true_neg_num == freqs_model_neg_num) else 0
            )

            sample_metrics.append(sample_data)
            n_samples += 1

            # Memory management
            torch.cuda.empty_cache()

            if max_samples is not None and n_samples >= max_samples:
                break

        # Create DataFrame from collected metrics
        df_results = pd.DataFrame(sample_metrics)

        # Save DataFrame
        df_results.to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")

    aggregated_results = {
        "energy_mae": df_results["energy_error"].mean(),
        "forces_mae": df_results["forces_error"].mean(),
        "hessian_mae": df_results["hessian_error"].mean(),
        "asymmetry_mae": df_results["asymmetry_error"].mean(),
        "true_asymmetry_mae": df_results["true_asymmetry_error"].mean(),
        "eigval_mae": df_results["eigval_mae"].mean(),
        "eigval1_mae": df_results["eigval1_mae"].mean(),
        "eigval2_mae": df_results["eigval2_mae"].mean(),
        "eigvec1_mae": df_results["eigvec1_mae"].mean(),
        "eigvec2_mae": df_results["eigvec2_mae"].mean(),
        "eigvec1_cos": df_results["eigvec1_cos"].mean(),
        "eigvec2_cos": df_results["eigvec2_cos"].mean(),
    }

    # Frequencies
    aggregated_results["neg_num_agree"] = df_results["neg_num_agree"].mean()
    aggregated_results["true_neg_num"] = df_results["true_neg_num"].mean()
    aggregated_results["model_neg_num"] = df_results["model_neg_num"].mean()
    aggregated_results["true_is_ts"] = df_results["true_is_ts"].mean()
    aggregated_results["model_is_ts"] = df_results["model_is_ts"].mean()
    aggregated_results["is_ts_agree"] = (
        df_results["model_is_ts"] == df_results["true_is_ts"]
    ).mean()

    print(f"\nResults for {dataset_name}:")
    print(f"Energy MAE: {aggregated_results['energy_mae']:.6f}")
    print(f"Forces MAE: {aggregated_results['forces_mae']:.6f}")
    print(f"Hessian MAE: {aggregated_results['hessian_mae']:.6f}")
    print(f"Asymmetry MAE: {aggregated_results['asymmetry_mae']:.6f}")
    print(f"True Asymmetry MAE: {aggregated_results['true_asymmetry_mae']:.6f}")
    print(f"Eigenvalue MAE: {aggregated_results['eigval_mae']:.6f} eV/Angstrom^2")
    print(f"Eigenvalue 1 MAE: {aggregated_results['eigval1_mae']:.6f}")
    print(f"Eigenvalue 2 MAE: {aggregated_results['eigval2_mae']:.6f}")
    print(f"Eigenvector 1 MAE: {aggregated_results['eigvec1_mae']:.6f}")
    print(f"Eigenvector 2 MAE: {aggregated_results['eigvec2_mae']:.6f}")
    print(f"Eigenvector 1 Cosine: {aggregated_results['eigvec1_cos']:.6f}")
    print(f"Eigenvector 2 Cosine: {aggregated_results['eigvec2_cos']:.6f}")

    # Frequencies
    print(f"True Neg Num: {aggregated_results['true_neg_num']:.6f}")
    print(f"Model Neg Num: {aggregated_results['model_neg_num']:.6f}")
    print(f"Neg Num Agree: {aggregated_results['neg_num_agree']:.6f}")
    print(f"True Is TS: {aggregated_results['true_is_ts']:.6f}")
    print(f"Model Is TS: {aggregated_results['model_is_ts']:.6f}")
    print(f"Is TS Agree: {aggregated_results['is_ts_agree']:.6f}")

    wandb.log(aggregated_results)

    if wandb_run_id is None:
        wandb.finish()

    return df_results, aggregated_results


def plot_accuracy_vs_natoms(df_results, name):
    """Plot accuracy metrics over number of atoms"""

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 10))
    fig.suptitle("Model Accuracy vs Number of Atoms", fontsize=16)

    # Define metrics to plot and their labels
    metrics = [
        ("energy_error", "Energy MAE", "Energy Error"),
        ("forces_error", "Forces MAE", "Forces Error"),
        ("hessian_error", "Hessian MAE", "Hessian Error"),
        ("eigvec1_cos", "Eigenvector 1 Cosine", "Eigenvector 1 Cosine"),
        ("eigval1_mae", "Eigenvalue 1 MAE", "Eigenvalue 1 MAE"),
        ("is_ts_agree", "Is TS Agree", "Is TS Agree"),
        ("neg_num_agree", "Neg Num Agree", "Neg Num Agree"),
        ("true_is_ts", "True Is TS", "True Is TS"),
        ("model_is_ts", "Model Is TS", "Model Is TS"),
    ]

    # Plot each metric
    for i, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[i // 2, i % 2]

        # Skip metrics not available in results
        if metric not in df_results.columns:
            ax.set_visible(False)
            continue

        # Group by natoms and calculate mean and std
        grouped = (
            df_results.groupby("natoms")[metric].agg(["mean", "std"]).reset_index()
        )

        # Plot mean with error bars
        ax.errorbar(
            grouped["natoms"],
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        ax.set_xlabel("Number of Atoms")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Set log scale for y-axis if needed (based on data range)
        if grouped["mean"].max() / grouped["mean"].min() > 100:
            ax.set_yscale("log")

    plt.tight_layout()

    # Save plot
    plot_dir = "plots/eval_horm"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f"{plot_dir}/accuracy_vs_natoms_{name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_filename}")

    # Show plot
    plt.show()
