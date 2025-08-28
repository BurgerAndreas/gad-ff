from typing import Dict, List, Optional, Tuple
from omegaconf import ListConfig
import yaml
import os
import torch
import warnings

# Suppress the FutureWarning from e3nn about torch.load weights_only parameter
warnings.filterwarnings("ignore", category=FutureWarning, module="e3nn")

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
from gadff.hessian_eigen import (
    projector_vibrational_modes,
    get_modes_geometric,
    compute_cartesian_modes,
    compute_vibrational_modes,
)
from ocpmodels.ff_lmdb import LmdbDataset
from ocpmodels.hessian_graph_transform import HessianGraphTransform


def get_model(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    # model_config["otf_graph"] = False
    print("model_config", model_config)
    return EquiformerV2_OC20(**model_config), model_config


def get_model_and_dataloader_for_hessian_prediction(
    batch_size,
    shuffle,
    device,
    dataset_path=None,
    config_path=None,
    checkpoint_path=None,
    dataloader_kwargs={},
):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Model
    if config_path is None:
        config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    model_config["do_hessian"] = True
    model_config["otf_graph"] = False
    model = EquiformerV2_OC20(**model_config)
    # Checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
    state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
    state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.train()
    model.to(device)
    # Dataset
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    if dataset_path is None:
        dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    dataset = LmdbDataset(dataset_path, transform=transform)
    # Dataloader
    follow_batch = ["diag_ij", "edge_index", "message_idx_ij"]
    dataloader = TGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=follow_batch,
        **dataloader_kwargs,
    )
    return model, dataloader


# https://github.com/deepprinciple/HORM/blob/eval/eval.py
def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    """Helper function to compute derivatives"""
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, forces, retain_graph=True):
    """Compute Hessian matrix using autograd."""

    # Get number of components (n_atoms * 3)
    n_comp = forces.reshape(-1).shape[0]

    # Initialize hessian
    hess = []
    for f in forces.reshape(-1):
        # Compute second-order derivative for each element
        hess_row = _get_derivatives(coords, -f, retain_graph=retain_graph)
        hess.append(hess_row)

    # Stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


class EquiformerTorchCalculator:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        eigen_dof_method: str = None,
        # pass in a model directly to avoid loading another model into memory
        model: torch.nn.Module = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model is None:
            project_root = os.path.dirname(os.path.dirname(__file__))

            config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
            self.model, self.model_config = get_model(config_path)

            if checkpoint_path is None:
                checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
            state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
            state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = model

        self.model.eval()
        self.model.to(device)

        # Method to compute the eigenvectors of the Hessian
        # qr: QR-based projector method (default)
        # svd: SVD-based projector method
        # inertia: Inertia tensor-based projector with auto-linearity detection
        # geo: Use Geometric library (external dependency)
        # ase: Use ASE library (external dependency)
        # eckart: Eckart frame alignment with principal axes
        self.eigen_dof_method = eigen_dof_method

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

    def get_forces(self, batch):
        """Get forces from the model"""
        batch = batch.to(self.model.device)
        batch = compute_extra_props(batch, pos_require_grad=False)
        _, forces, _ = self.model.forward(batch, eigen=False)
        return forces

    def get_energy(self, batch):
        """Get energy from the model"""
        batch = batch.to(self.model.device)
        batch = compute_extra_props(batch, pos_require_grad=False)
        energy, _, _ = self.model.forward(batch, eigen=False)
        return energy

    def predict_with_hessian(self, batch):
        """Predict one batch with autodiff Hessian"""
        B = batch.batch.max() + 1
        assert B == 1, "Only one batch is supported for Hessian prediction"
        N = batch.natoms

        # Prepare batch with extra properties
        batch = batch.to(self.model.device)
        batch = compute_extra_props(batch, pos_require_grad=True)

        # Run prediction
        with torch.enable_grad():
            energy, forces, eigenpred = self.model.forward(batch, eigen=True)

        # 3D coordinates -> 3N^2 Hessian elements
        hessian = compute_hessian(batch.pos, forces, retain_graph=True)

        # A named tuple (eigenvalues, eigenvectors) which corresponds to Λ and Q in: M = Qdiag(Λ)Q^T
        # eigenvalues will always be real-valued. It will also be ordered in ascending order.
        # eigenvectors contain the eigenvectors as its columns.
        # eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        eigenvalues, eigenvectors = compute_vibrational_modes(
            hessian,
            batch.z,
            batch.pos,
            forces=forces,
            method=self.eigen_dof_method,
        )
        smallest_eigenvals = eigenvalues[:2]
        smallest_eigenvecs = eigenvectors[:, :2]  # [N*3, 2]
        eigenvalues = smallest_eigenvals
        eigenvectors = smallest_eigenvecs.T.view(2, N, 3)
        return energy, forces, hessian, eigenvalues, eigenvectors, eigenpred

    def predict_gad(self, batch):
        """
        Gentlest Ascent Dynamics (GAD)
        dx/dt = -∇V(x) + 2(∇V, v(x))v(x)
        = F + 2(-F, v(x))v(x)
        since F=-∇V(x)
        where v(x) is the eigenvector of the Hessian with the smallest eigenvalue.
        """
        B = batch.batch.max() + 1
        energy, forces, eigenpred = self.predict(batch)
        v = eigenpred["eigvec_1"].reshape(B, -1)
        # normalize eigenvector
        v = v / torch.norm(v, dim=1, keepdim=True)
        forces = forces.reshape(B, -1)
        # −∇V(x) + 2(∇V, v(x))v(x)
        gad = forces + 2 * torch.einsum("bi,bi->b", -forces, v) * v
        out = {
            "energy": energy,
            "forces": forces,
        }
        out.update(eigenpred)
        return gad, out

    def gad_autograd_hessian(self, batch):
        B = batch.batch.max() + 1
        assert B == 1, "Only one batch is supported for Hessian prediction"
        N = batch.natoms

        # Prepare batch with extra properties
        batch = batch.to(self.model.device)
        batch = compute_extra_props(batch, pos_require_grad=True)

        # Run prediction
        with torch.enable_grad():
            energy, forces, eigenpred = self.model.forward(batch, eigen=False)

        hessian = compute_hessian(batch.pos, forces, retain_graph=True)

        # A named tuple (eigenvalues, eigenvectors) which corresponds to Λ and Q in: M = Qdiag(Λ)Q^T
        # eigenvalues will always be real-valued. It will also be ordered in ascending order.
        # eigenvectors contain the eigenvectors as its columns.
        # eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        # eigenvalues, eigenvectors = projector_vibrational_modes(
        #     pos=batch.pos,
        #     atom_types=batch.z,
        #     H=hessian,
        # )
        # eigenvalues, eigenvectors = get_modes_geometric(
        #     hessian, batch.z, batch.pos, True, True
        # )
        eigenvalues, eigenvectors = compute_vibrational_modes(
            hessian,
            batch.z,
            batch.pos,
            forces=forces,
            method=self.eigen_dof_method,
        )

        eigenvalues = eigenvalues[:2]
        eigenvectors = eigenvectors[:, :2]
        v = eigenvectors[:, 0].reshape(-1)  # [N*3]

        # eigenvector should be normalized anyway
        v = v.to(dtype=forces.dtype)
        v = v / torch.norm(v)

        forces = forces.reshape(-1)  # [N*3]

        # dx/dt = -∇V(x) + 2(∇V, v(x))v(x)
        # = F + 2(-F, v(x))v(x)
        # since F = -∇V(x)
        dot_product = torch.dot(-forces, v)
        gad = forces + (2 * dot_product * v)
        out = {
            "energy": energy,
            "forces": forces,
            "hessian": hessian,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }
        out.update(eigenpred)
        return gad, out

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
