
from typing import Dict, List, Optional, Tuple
from omegaconf import ListConfig
import os
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

import yaml

GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8])

def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x

def compute_extra_props(batch, pos_require_grad=True):
    """Adds device, z, and removes mean batch"""
    device = batch.pos.device
    indices = batch.one_hot.long().argmax(dim=1)
    batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    # atomization energy. shape used by equiformerv2
    if not hasattr(batch, "ae"):
        batch.ae = torch.zeros_like(batch.energy)
    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch

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
    batch = compute_extra_props(
        batch, pos_require_grad=True
    )

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
    hessian = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=None)(I_N)[
        0
    ]
    hessian = hessian.view(N * 3, N * 3)

    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    smallest_eigenvals = eigenvalues[:2]
    smallest_eigenvecs = eigenvectors[:, :2]
    eigenvalues = smallest_eigenvals
    eigenvectors = (
        smallest_eigenvecs.T.view(2, N, 3)
    )
    return energy, forces, hessian, eigenvalues, eigenvectors, eigenpred

def predict_gad(batch, potential):
    B = batch.batch.max() + 1
    energy, forces, eigenpred = predict(batch, potential)
    v = eigenpred["eigvec_1"].reshape(B, -1)
    forces = forces.reshape(B, -1)
    # −∇V(x) + 2(∇V, v(x))v(x)
    gad = -forces + 2 * torch.einsum("bi,bi->b", forces, v) * v
    return gad

def predict_gad_with_hessian(batch, potential):
    energy, forces, hessian, eigenvalues, eigenvectors, eigenpred = predict_with_hessian(batch, potential)
    v = eigenvectors[0].reshape(-1) # N*3
    forces = forces.reshape(-1) # N*3
    # −∇V(x) + 2(∇V, v(x))v(x)
    gad = -forces + 2 * torch.einsum("i,i->", forces, v) * v
    return gad

if __name__ == "__main__":
    from torch_geometric.data import Data as TGData
    from torch_geometric.loader import DataLoader as TGDataLoader
    
    # you might need to change this
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
    model, model_config = get_model(config_path)
    
    checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True)["state_dict"], 
        strict=False
    )
    
    model.eval()
    model.to("cuda")
    
    # Example 1: load a dataset file and predict the first batch
    # from data.ff_lmdb import LmdbDataset
    from data.ff_lmdb import LmdbDataset
    dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    dataset = LmdbDataset(dataset_path)
    # either use the dataset directly or use a dataloader
    batch = dataset[0]
    batch = Batch.from_data_list([batch])
    # dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)
    # batch = next(iter(dataloader))
    energy, forces, eigenpred = predict(batch, model)
    print(f"Energy: {energy.shape}")
    print(f"Forces: {forces.shape}")
    print(f"Eigenpred: {eigenpred.keys()}")
    
    # Example 2: create a random data object with random positions and predict
    n_atoms = 10
    data = TGData(
        pos=torch.randn(n_atoms, 3),
        batch=torch.zeros(n_atoms),
        one_hot=torch.randint(0, 4, (n_atoms, 4)),
        natoms=n_atoms,
        # just needs be a placeholder that decides the output energy shape
        energy=torch.randn(1), 
        # forces=torch.randn(n_atoms, 3),
        # ae=torch.zeros(1),
    )
    data = Batch.from_data_list([data])
    
    energy, forces, eigenpred = predict(data, model)
    print(f"Energy: {energy.shape}")
    print(f"Forces: {forces.shape}")
    print(f"Eigenpred: {eigenpred.keys()}")
    
    # Example 3: predict gad
    gad = predict_gad(data, model)
    print(f"GAD: {gad.shape}")
    
    # Example 4: predict gad with hessian
    gad = predict_gad_with_hessian(data, model)
    print(f"GAD with Hessian: {gad.shape}")
    
    