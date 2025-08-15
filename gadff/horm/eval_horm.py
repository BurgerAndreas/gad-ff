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


def evaluate(
    lmdb_path,
    checkpoint_path,
    config_path,
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
            project="reactbench",
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

    # if hessian_method == "predict" or model.do_hessian or model.otf_graph == False:
    if hessian_method == "predict":
        transform = HessianGraphTransform(
            cutoff=model.cutoff,
            max_neighbors=model.max_neighbors,
            use_pbc=model.use_pbc,
        )
    else:
        transform = None

    dataset = LmdbDataset(fix_dataset_path(lmdb_path), transform=transform)
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    dataset_name = lmdb_path.split("/")[-1].split(".")[0]
    
    # Create results file path
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    ckpt_name = checkpoint_path.split("/")[-1].split(".")[0]
    results_file = f"{results_dir}/{ckpt_name}_{dataset_name}_{hessian_method}_metrics.pkl"
    
    # Check if results already exist and redo is False
    if os.path.exists(results_file) and not redo:
        print(f"Loading existing results from {results_file}")
        df_results = pd.read_pickle(results_file)
        
    else:
        # Initialize metrics collection for per-sample DataFrame
        sample_metrics = []
        n_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            batch = batch.to("cuda")
            batch = compute_extra_props(batch)

            # Forward pass
            if model_name == "LEFTNet":
                batch.pos.requires_grad_()
                ener, force = model.forward_autograd(batch)
                hess = compute_hessian(batch.pos, ener, force)
            elif "equiformer" in model_name.lower():
                if do_autograd:
                    batch.pos.requires_grad_()
                    ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
                    hess = compute_hessian(batch.pos, ener, force)
                else:
                    ener, force, out = model.forward(batch, otf_graph=False, hessian=True)
                    hess = out["hessian"]
            else:
                batch.pos.requires_grad_()
                ener, force, out = model.forward(batch)
                hess = compute_hessian(batch.pos, ener, force)

            # Compute hessian eigenspectra
            eigvals, eigvecs = torch.linalg.eigh(hess)
            eigenvalues_hartree_bohr = hess2eigenvalues(hess)

            # Compute errors
            e_error = torch.mean(torch.abs(ener.squeeze() - batch.ae))
            f_error = torch.mean(torch.abs(force - batch.forces))

            # Reshape true hessian
            n_atoms = batch.pos.shape[0]
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
            h_error = torch.mean(torch.abs(hess - hessian_true))

            # Eigenvalue error
            eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)
            eigen_true_hartree_bohr = hess2eigenvalues(hessian_true)
            eigen_error = torch.mean(
                torch.abs(eigenvalues_hartree_bohr - eigen_true_hartree_bohr)
            )  # Hartree/Bohr^2

            # Asymmetry error
            asymmetry_error = torch.mean(torch.abs(hess - hess.T))
            true_asymmetry_error = torch.mean(torch.abs(hessian_true - hessian_true.T))

            # Additional metrics
            eigval_mae = torch.mean(torch.abs(eigvals - eigvals_true)) # eV/Angstrom^2
            eigval1_mae = torch.mean(torch.abs(eigvals[0] - eigvals_true[0]))
            eigval2_mae = torch.mean(torch.abs(eigvals[1] - eigvals_true[1]))
            eigvec1_mae = torch.mean(torch.abs(eigvecs[:, 0] - eigvecs_true[:, 0]))
            eigvec2_mae = torch.mean(torch.abs(eigvecs[:, 1] - eigvecs_true[:, 1]))
            eigvec1_cos = torch.abs(torch.dot(eigvecs[:, 0], eigvecs_true[:, 0]))
            eigvec2_cos = torch.abs(torch.dot(eigvecs[:, 1], eigvecs_true[:, 1]))
            
            # Collect per-sample metrics
            sample_data = {
                'sample_idx': n_samples,
                'natoms': n_atoms,
                'energy_error': e_error.item(),
                'forces_error': f_error.item(),
                'hessian_error': h_error.item(),
                'eigen_error': eigen_error.item(),
                'asymmetry_error': asymmetry_error.item(),
                'true_asymmetry_error': true_asymmetry_error.item(),
                'eigval_mae': eigval_mae.item(),
                'eigval1_mae': eigval1_mae.item(),
                'eigval2_mae': eigval2_mae.item(),
                'eigvec1_mae': eigvec1_mae.item(),
                'eigvec2_mae': eigvec2_mae.item(),
                'eigvec1_cos': eigvec1_cos.item(),
                'eigvec2_cos': eigvec2_cos.item(),
            }
            sample_metrics.append(sample_data)
            n_samples += 1

            # Memory management
            torch.cuda.empty_cache()

            if max_samples is not None and n_samples >= max_samples:
                break

        # Create DataFrame from collected metrics
        df_results = pd.DataFrame(sample_metrics)
        
        # Save DataFrame
        df_results.to_pickle(results_file)
        print(f"Saved results to {results_file}")
        
    aggregated_results = {
        "energy_mae": df_results["energy_error"].mean(),
        "forces_mae": df_results["forces_error"].mean(), 
        "hessian_mae": df_results["hessian_error"].mean(),
        "eigenvalue_mae_hartree_bohr2": df_results["eigen_error"].mean(),
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
    
    print(f"\nResults for {dataset_name}:")
    print(f"Energy MAE: {aggregated_results['energy_mae']:.6f}")
    print(f"Forces MAE: {aggregated_results['forces_mae']:.6f}")
    print(f"Hessian MAE: {aggregated_results['hessian_mae']:.6f}")
    print(f"Eigenvalue MAE: {aggregated_results['eigenvalue_mae_hartree_bohr2']:.6f} Hartree/Bohr^2")
    print(f"Asymmetry MAE: {aggregated_results['asymmetry_mae']:.6f}")
    print(f"True Asymmetry MAE: {aggregated_results['true_asymmetry_mae']:.6f}")
    print(f"Eigenvalue MAE: {aggregated_results['eigval_mae']:.6f} eV/Angstrom^2")
    print(f"Eigenvalue 1 MAE: {aggregated_results['eigval1_mae']:.6f}")
    print(f"Eigenvalue 2 MAE: {aggregated_results['eigval2_mae']:.6f}")
    print(f"Eigenvector 1 MAE: {aggregated_results['eigvec1_mae']:.6f}")
    print(f"Eigenvector 2 MAE: {aggregated_results['eigvec2_mae']:.6f}")
    print(f"Eigenvector 1 Cosine: {aggregated_results['eigvec1_cos']:.6f}")
    print(f"Eigenvector 2 Cosine: {aggregated_results['eigvec2_cos']:.6f}")
    
    wandb.log(aggregated_results)

    if wandb_run_id is None:
        wandb.finish()

    return df_results, aggregated_results


def plot_accuracy_vs_natoms(df_results):
    """Plot accuracy metrics over number of atoms"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Accuracy vs Number of Atoms', fontsize=16)
    
    # Define metrics to plot and their labels
    metrics = [
        ('energy_error', 'Energy MAE', 'Energy Error'),
        ('forces_error', 'Forces MAE', 'Forces Error'),
        ('hessian_error', 'Hessian MAE', 'Hessian Error'),
        ('eigen_error', 'Eigenvalue MAE', 'Eigenvalue Error (Hartree/Bohr²)')
    ]
    
    # Plot each metric
    for i, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # Group by natoms and calculate mean and std
        grouped = df_results.groupby('natoms')[metric].agg(['mean', 'std']).reset_index()
        
        # Plot mean with error bars
        ax.errorbar(grouped['natoms'], grouped['mean'], yerr=grouped['std'], 
                   marker='o', capsize=5, capthick=2, linewidth=2)
        
        ax.set_xlabel('Number of Atoms')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Set log scale for y-axis if needed (based on data range)
        if grouped['mean'].max() / grouped['mean'].min() > 100:
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = "accuracy_vs_natoms.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")
    
    # Show plot
    plt.show()
