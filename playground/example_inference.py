from typing import Dict, List, Optional, Tuple
from omegaconf import ListConfig
import os
import torch
from torch_geometric.data import Batch
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props
import yaml

# TODO: not up to date!


def get_model(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    return EquiformerV2_OC20(**model_config), model_config


def predict(batch, potential):
    """Predict one or multiple batches"""
    batch = batch.to(potential.device)
    batch = compute_extra_props(batch, pos_require_grad=False)
    energy, forces, eigenpred = potential.forward(batch, eigen=True)
    return energy, forces, eigenpred


def predict_with_hessian(batch, potential):
    """Predict one batch with autodiff Hessian"""
    B = batch.batch.max() + 1
    assert B == 1, "Only one batch is supported for Hessian prediction"

    batch = batch.to(potential.device)

    # Prepare batch with extra properties
    batch = compute_extra_props(batch, pos_require_grad=True)

    # Run prediction
    with torch.enable_grad():
        energy, forces, eigenpred = potential.forward(batch, eigen=True)

    # 3D coordinates -> 3N^2 Hessian elements
    N = batch.pos.shape[0]
    forces = forces.reshape(-1)
    num_elements = forces.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            outputs=-1 * forces,
            inputs=batch.pos,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements, device=forces.device)
    hessian = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=None)(I_N)[0]
    hessian = hessian.view(N * 3, N * 3)

    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    smallest_eigenvals = eigenvalues[:2]
    smallest_eigenvecs = eigenvectors[:, :2]
    eigenvalues = smallest_eigenvals
    eigenvectors = smallest_eigenvecs.T.view(2, N, 3)
    return energy, forces, hessian, eigenvalues, eigenvectors, eigenpred


def predict_gad(batch, potential):
    B = batch.batch.max() + 1
    energy, forces, eigenpred = predict(batch, potential)
    v = eigenpred["eigvec_1"].reshape(B, -1)
    v = v / torch.norm(v, dim=1, keepdim=True)
    forces = forces.reshape(B, -1)
    # -∇V(x) + 2(∇V, v(x))v(x)
    gad = forces + 2 * torch.einsum("bi,bi->b", -forces, v) * v
    return gad


def gad_autograd_hessian(batch, potential):
    energy, forces, hessian, eigenvalues, eigenvectors, eigenpred = (
        predict_with_hessian(batch, potential)
    )
    v = eigenvectors[0].reshape(-1)  # N*3
    v = v / torch.norm(v, dim=0, keepdim=True)
    forces = forces.reshape(-1)  # N*3
    # -∇V(x) + 2(∇V, v(x))v(x)
    gad = forces + 2 * torch.einsum("i,i->", -forces, v) * v
    return gad


if __name__ == "__main__":
    from torch_geometric.data import Data as TGData
    from gadff.inference_utils import get_model_from_checkpoint, get_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # you might need to change this
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "ckpt/hesspred_v1.ckpt")
    model = get_model_from_checkpoint(checkpoint_path, device)

    # Example 1: load a dataset file and predict the first batch
    dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    dataloader = get_dataloader(dataset_path, model, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    energy, forces, eigenpred = predict(batch, model)
    print("\nExample 1:")
    print(f"  Energy: {energy.shape}")
    print(f"  Forces: {forces.shape}")
    print(f"  Eigenpred: {eigenpred.keys()}")

    # Example 2: create a random data object with random positions and predict
    n_atoms = 10
    elements = torch.tensor([1, 6, 7, 8])  # H, C, N, O
    data = TGData(
        pos=torch.randn(n_atoms, 3),
        z=elements[torch.randint(0, 4, (n_atoms,))],
        natoms=n_atoms,
    )
    data = Batch.from_data_list([data])

    energy, forces, eigenpred = predict(data, model)
    print("\nExample 2:")
    print(f"  Energy: {energy.shape}")
    print(f"  Forces: {forces.shape}")
    print(f"  Eigenpred: {eigenpred.keys()}")

    # Example 3: predict gad
    gad = predict_gad(data, model)
    print("\nExample 3:")
    print(f"  GAD: {gad.shape}")

    # Example 4: predict gad with hessian
    gad = gad_autograd_hessian(data, model)
    print(f"  GAD with Hessian: {gad.shape}")
