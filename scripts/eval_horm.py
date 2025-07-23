from gadff.horm.training_module import PotentialModule, compute_extra_props
from torch_geometric.loader import DataLoader
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path
import torch
from tqdm import tqdm


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


def evaluate(lmdb_path, checkpoint_path, max_samples=None):

    ckpt = torch.load(checkpoint_path)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]

    print(f"Model name: {model_name}")

    pm = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to("cuda")

    dataset = LmdbDataset(fix_dataset_path(lmdb_path))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    dataset_name = lmdb_path.split("/")[-1].split(".")[0]

    # Initialize metrics
    total_e_error = 0.0
    total_f_error = 0.0
    total_h_error = 0.0
    total_eigen_error = 0.0
    total_asymmetry_error = 0.0
    n_samples = 0
    
    # Added Andreas
    total_eigval1_mae = 0.0
    total_eigval2_mae = 0.0
    total_eigvec1_mae = 0.0
    total_eigvec2_mae = 0.0
    total_eigvec1_cos = 0.0
    total_eigvec2_cos = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):

        batch = batch.to("cuda")
        batch.pos.requires_grad_()
        batch = compute_extra_props(batch)

        # Forward pass
        if model_name == "LEFTNet":
            ener, force = pm.forward_autograd(batch)
        else:
            ener, force, out = pm.forward(batch)

        # Compute hessian and eigenvalues
        # Use reshape instead of view to handle non-contiguous tensors
        hess = compute_hessian(batch.pos, ener, force)
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
        eigen_error = torch.mean(torch.abs(eigenvalues_hartree_bohr - eigen_true_hartree_bohr))

        # Asymmetry error
        asymmetry_error = torch.mean(torch.abs(hess - hess.T))
        total_asymmetry_error += asymmetry_error.item()

        # Update totals
        total_e_error += e_error.item()
        total_f_error += f_error.item()
        total_h_error += h_error.item()
        total_eigen_error += eigen_error.item()
        n_samples += 1
        
        # Added Andreas
        eigval1_mae = torch.mean(torch.abs(eigvals[0] - eigvals_true[0]))
        eigval2_mae = torch.mean(torch.abs(eigvals[1] - eigvals_true[1]))
        eigvec1_mae = torch.mean(torch.abs(eigvecs[:, 0] - eigvecs_true[:, 0]))
        eigvec2_mae = torch.mean(torch.abs(eigvecs[:, 1] - eigvecs_true[:, 1]))
        eigvec1_cos = torch.abs(torch.dot(eigvecs[:, 0], eigvecs_true[:, 0]))
        eigvec2_cos = torch.abs(torch.dot(eigvecs[:, 1], eigvecs_true[:, 1]))
        total_eigval1_mae += eigval1_mae.item()
        total_eigval2_mae += eigval2_mae.item()
        total_eigvec1_mae += eigvec1_mae.item()
        total_eigvec2_mae += eigvec2_mae.item()
        total_eigvec1_cos += eigvec1_cos.item()
        total_eigvec2_cos += eigvec2_cos.item()

        # Memory management
        torch.cuda.empty_cache()
        
        if max_samples is not None and n_samples >= max_samples:
            break

    # Calculate average errors
    mae_e = total_e_error / n_samples
    mae_f = total_f_error / n_samples
    mae_h = total_h_error / n_samples
    mae_eigen = total_eigen_error / n_samples
    mae_asymmetry = total_asymmetry_error / n_samples

    print(f"\nResults for {dataset_name}:")
    print(f"Energy MAE: {mae_e:.6f}")
    print(f"Forces MAE: {mae_f:.6f}")
    print(f"Hessian MAE: {mae_h:.6f}")
    print(f"Eigenvalue MAE: {mae_eigen:.6f}")
    print(f"Asymmetry MAE: {mae_asymmetry:.6f}")
    
    # Added Andreas
    print(f"Eigenvalue 1 MAE: {total_eigval1_mae / n_samples:.6f}")
    print(f"Eigenvalue 2 MAE: {total_eigval2_mae / n_samples:.6f}")
    print(f"Eigenvector 1 MAE: {total_eigvec1_mae / n_samples:.6f}")
    print(f"Eigenvector 2 MAE: {total_eigvec2_mae / n_samples:.6f}")
    print(f"Eigenvector 1 Cosine: {total_eigvec1_cos / n_samples:.6f}")
    print(f"Eigenvector 2 Cosine: {total_eigvec2_cos / n_samples:.6f}")


if __name__ == "__main__":

    torch.manual_seed(42)

    checkpoint_path = "ckpt/eqv2.ckpt"

    DATASET_FILES_HORM = [
        "ts1x-val.lmdb",  # 50844 samples
        "ts1x_hess_train_big.lmdb",  # 1725362 samples
        "RGD1.lmdb",  # 60000 samples
    ]
    # lmdb_path = 'data/sample_100.lmdb'
    lmdb_path = DATASET_FILES_HORM[0]

    # evaluate(lmdb_path, checkpoint_path, max_samples=1000)
    evaluate(lmdb_path, checkpoint_path)
