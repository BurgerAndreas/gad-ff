from typing import Dict, List, Optional, Tuple
from omegaconf import ListConfig
import yaml
import os
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
from torch_geometric.loader import DataLoader as TGDataLoader

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props
from ocpmodels.common.relaxation.ase_utils import (
    batch_to_atoms,
    ase_atoms_to_torch_geometric,
)
from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs

from ase.calculators.calculator import Calculator
from ase import Atoms


def get_model(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    return EquiformerV2_OC20(**model_config), model_config


class EquiformerCalculator:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        project_root = os.path.dirname(os.path.dirname(__file__))

        config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
        self.model, self.model_config = get_model(config_path)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True)["state_dict"], strict=False
        )

        self.model.eval()
        self.model.to(device)

        # ocpmodels/common/relaxation/ase_utils.py
        self.a2g = AtomsToGraphs(
            max_neigh=self.model.max_neighbors,
            radius=self.model.cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )

    def predict(self, batch):
        """Predict one or multiple batches"""
        batch = batch.to(self.model.device)
        batch = compute_extra_props(batch, pos_require_grad=False)
        energy, forces, eigenpred = self.model.forward(batch, eigen=True)
        return energy, forces, eigenpred

    def predict_with_hessian(self, batch):
        """Predict one batch with autodiff Hessian"""
        B = batch.batch.max() + 1
        assert B == 1, "Only one batch is supported for Hessian prediction"

        batch = batch.to(self.model.device)

        # Prepare batch with extra properties
        batch = compute_extra_props(batch, pos_require_grad=True)

        # Run prediction
        with torch.enable_grad():
            energy, forces, eigenpred = self.model.forward(batch, eigen=True)

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

    def predict_gad(self, batch):
        B = batch.batch.max() + 1
        energy, forces, eigenpred = self.predict(batch)
        v = eigenpred["eigvec_1"].reshape(B, -1)
        forces = forces.reshape(B, -1)
        # −∇V(x) + 2(∇V, v(x))v(x)
        gad = -forces + 2 * torch.einsum("bi,bi->b", forces, v) * v
        return gad

    def predict_gad_with_hessian(self, batch):
        energy, forces, hessian, eigenvalues, eigenvectors, eigenpred = (
            self.predict_with_hessian(batch)
        )
        v = eigenvectors[0].reshape(-1)  # N*3
        forces = forces.reshape(-1)  # N*3
        # −∇V(x) + 2(∇V, v(x))v(x)
        gad = -forces + 2 * torch.einsum("i,i->", forces, v) * v
        return gad

    def find_transitionstate_with_gad_from_hessian(self, batch):
        """Integrate the equations of motion of the GAD vector field."""
        raise NotImplementedError("Not implemented")

    def ase_to_batch(self, atoms: Atoms):
        # Call base class to set atoms attribute
        Calculator.calculate(self, atoms)

        # ocpmodels/common/relaxation/ase_utils.py
        # data_object = self.a2g.convert(atoms)
        # batch = data_list_collater([data_object], otf_graph=True)

        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric(atoms)
        return batch
