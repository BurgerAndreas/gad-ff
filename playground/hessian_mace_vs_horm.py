import copy
from typing import List, Optional, Tuple
import time
import os
import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn
import torch.utils.data

from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props

###################################################################################
# MACE


# https://github.com/ACEsuit/mace/blob/1d5b6a0bdfdc7258e0bb711eda1c998a4aa77976/mace/modules/utils.py#L112
@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except Exception as e:
        print("RuntimeError in compute_hessians_vmap", e)
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


###################################################################################
# HORM
# https://github.com/deepprinciple/HORM/blob/eval/eval.py


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
    """Convert Hessian to eigenvalues with proper unit conversion"""
    hartree_to_ev = 27.2114
    bohr_to_angstrom = 0.529177
    ev_angstrom_2_to_hartree_bohr_2 = (bohr_to_angstrom**2) / hartree_to_ev

    hess = hess * ev_angstrom_2_to_hartree_bohr_2
    eigen_values, _ = torch.linalg.eigh(hess)
    return eigen_values


if __name__ == "__main__":

    # you might need to change this
    project_root = os.path.dirname(os.path.dirname(__file__))

    config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    model = EquiformerV2_OC20(**model_config)

    checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
    state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
    state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to("cuda")

    # Example 1: load a dataset file and predict the first batch
    from ocpmodels.ff_lmdb import LmdbDataset

    dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    dataset = LmdbDataset(dataset_path)

    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    print("\n")
    for batch_base in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):

        N = batch_base.natoms

        # MACE
        batch = batch_base.clone()
        batch = batch.to(model.device)
        batch = compute_extra_props(batch, pos_require_grad=True)
        energy, forces, eigenpred = model.forward(batch, eigen=True)
        forces = forces.reshape(-1)
        print("Forces", forces.shape)
        print("Pos", batch.pos.shape)
        hessian_mace = compute_hessians_vmap(forces, batch.pos)  # [N*3, N, 3]
        hessian_mace = hessian_mace.reshape(N * 3, N * 3)
        print("Hessian MACE", hessian_mace.shape)
        mace_eigval, mace_eigvec = torch.linalg.eigh(hessian_mace)
        forces1 = forces.clone().detach()

        # HORM
        # batch = batch_base.clone()
        # batch = batch.to(model.device)
        # batch = compute_extra_props(batch, pos_require_grad=True)
        # energy, forces, eigenpred = model.forward(batch, eigen=True)
        hessian_horm = compute_hessian(batch.pos, energy, forces)
        print("Horm Hessian", hessian_horm.shape)
        horm_eigval, horm_eigvec = torch.linalg.eigh(hessian_horm)

        print("")
        forces2 = forces.clone().detach()
        dff = forces1 - forces2.reshape(-1)
        print("Forces Max diff", dff.abs().max())
        print("Forces Mean diff", dff.abs().mean())

        dff = hessian_mace - hessian_horm
        print("Hessian Max diff", dff.abs().max())
        print("Hessian Mean diff", dff.abs().mean())

        # print("")
        # print(hessian_mace[0][:20])
        # print(hessian_horm[0][:20])

        print("")
        diff = horm_eigvec[:, 0] - mace_eigvec[:, 0]
        print("Eigenvector Max diff", diff.abs().max())
        print("Eigenvector Mean diff", diff.abs().mean())

        print("")
        diff = mace_eigval - horm_eigval
        print("Eigenvalue Max diff", diff.abs().max())
        print("Eigenvalue Mean diff", diff.abs().mean())

        break

    # Time the difference
    print("")
    start_time = time.time()
    for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
        batch = batch.to(model.device)
        batch = compute_extra_props(batch, pos_require_grad=True)
        N = batch.natoms
        energy, forces, eigenpred = model.forward(batch, eigen=True)
        forces = forces.reshape(-1)
        hessian_mace = compute_hessians_vmap(forces, batch.pos)  # [N*3, N, 3]
        hessian_mace = hessian_mace.reshape(N * 3, N * 3)
    end_time = time.time()
    print(f"MACE time taken: {end_time - start_time:.1f} seconds")

    start_time = time.time()
    for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
        batch = batch.to(model.device)
        batch = compute_extra_props(batch, pos_require_grad=True)
        energy, forces, eigenpred = model.forward(batch, eigen=True)
        hessian_horm = compute_hessian(batch.pos, energy, forces)  # [N*3, N, 3]
    end_time = time.time()
    print(f"Horm time taken: {end_time - start_time:.1f} seconds")
