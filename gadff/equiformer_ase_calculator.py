"""
ASE Calculator wrapper for Equiformer model.
"""

from typing import Optional
import numpy as np
import yaml
import os

import torch
from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.constraints import FixAtoms
from torch_scatter import scatter_mean

from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.common.relaxation.ase_utils import (
    batch_to_atoms,
    ase_atoms_to_torch_geometric,
)

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props
from gadff.hessian_eigen import projector_vibrational_modes


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


class EquiformerASECalculator(Calculator):
    """
    Equiformer ASE Calculator.

    Might need to reimplement EquiformerASECalculator based on:
    ocpmodels/common/relaxation/ase_utils.py

    Args:
        checkpoint_path: Path to the Equiformer model checkpoint
        device: Optional device specification (defaults to auto-detect)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        project_root: str = None,
        hessian_method: str = "autodiff",
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """
        Initialize the Equiformer calculator.

        Args:
            checkpoint_path: Path to the trained Equiformer checkpoint file
            device: Optional device specification (defaults to auto-detect)
            **kwargs: Additional keyword arguments for parent Calculator class
        """
        Calculator.__init__(self, **kwargs)

        # this is where all the calculated properties are stored
        self.results = {}

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model is None:
            # Load model
            if project_root is None:
                project_root = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            model_config = config["model"]
            self.potential = EquiformerV2_OC20(**model_config)

            if checkpoint_path is None:
                checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
            state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
            state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
            self.potential.load_state_dict(state_dict, strict=False)
        else:
            self.potential = model

        # self.potential = self.potential.to(self.device)
        self.potential.to(self.device)
        self.potential.eval()

        # Set implemented properties
        # # standard properties: ‘energy’, ‘forces’, ‘stress’, ‘dipole’, ‘charges’, ‘magmom’ and ‘magmoms’.
        self.implemented_properties = ["energy", "forces", "hessian"]

        self.hessian_method = hessian_method

        # ocpmodels/common/relaxation/ase_utils.py
        self.a2g = AtomsToGraphs(
            max_neigh=self.potential.max_neighbors,
            radius=self.potential.cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )

    def reset(self):
        """Reset the calculator."""
        self.results = {}

    def forward(self, atoms, hessian=False):
        """
        Forward pass for the Equiformer calculator.
        If hessian is True, it will compute the Hessian via autograd and eigenvalues/eigenvectors.
        Otherwise, it will only compute the energy and forces.
        """
        properties = ["energy", "forces"]
        if hessian:
            properties += ["hessian", "eigen"]
        self.calculate(atoms, properties=properties)
        return self.results

    def calculate(
        self,
        atoms=None,
        properties=None,
        hessian_method=None,
        system_changes=all_changes,
    ):
        """
        Calculate properties for the given atoms.

        You can get the

        Args:
            atoms: ASE Atoms object
            properties: List of properties to compute (used by ASE internally)
            system_changes: System changes since last calculation (used by ASE internally)
        """
        # Call base class to set atoms attribute
        Calculator.calculate(self, atoms)

        # ocpmodels/common/relaxation/ase_utils.py
        # data_object = self.a2g.convert(atoms)
        # batch = data_list_collater([data_object], otf_graph=True)

        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric(atoms)
        batch = batch.to(self.device)

        if properties is None:
            properties = []
        if "eigen" in properties:
            properties.append("hessian")

        if hessian_method is None:
            hessian_method = self.hessian_method

        # Prepare batch with extra properties
        batch = compute_extra_props(batch, pos_require_grad="hessian" in properties)

        # Run prediction
        if (hessian_method == "autodiff") and ("hessian" in properties):
            with torch.enable_grad():
                energy, forces, eigenoutputs = self.potential.forward(
                    batch, eigen=True, hessian=False
                )
        else:
            with torch.no_grad():
                energy, forces, eigenoutputs = self.potential.forward(
                    batch, eigen=True, hessian=hessian_method == "predict"
                )

        # Store results
        self.results = {}

        # Energy is per molecule, extract scalar value
        self.results["energy"] = float(energy.detach().cpu().item())

        # Forces shape: [n_atoms, 3]
        self.results["forces"] = forces.detach().cpu().numpy()

        # predicted eigenvalues and eigenvectors of the Hessian
        for key in ["eigval_1", "eigval_2", "eigvec_1", "eigvec_2"]:
            if key in eigenoutputs:
                self.results[key] = eigenoutputs[key].detach().cpu().numpy()

        # Compute the Hessian via autodiff on the fly
        if "hessian" in properties:
            if hessian_method == "autodiff":
                # 3D coordinates -> 3N^2 Hessian elements
                N = batch.pos.shape[0]
                forces = forces.reshape(-1)
                num_elements = forces.shape[0]

                hessian = compute_hessian(batch.pos, forces, retain_graph=True)
            elif hessian_method == "predict":
                hessian = eigenoutputs["hessian"]
            else:
                raise ValueError(f"Invalid Hessian method: {hessian_method}")
            self.results["hessian"] = hessian.detach().cpu().numpy()

        if "eigen" in properties:
            # eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            eigenvalues, eigenvectors = projector_vibrational_modes(
                pos=batch.pos,
                atom_types=batch.z,
                H=hessian,
            )
            smallest_eigenvals = eigenvalues[:2]
            smallest_eigenvecs = eigenvectors[:, :2]
            self.results["eigenvalues"] = smallest_eigenvals.detach().cpu().numpy()
            self.results["eigenvectors"] = (
                smallest_eigenvecs.T.view(2, N, 3).detach().cpu().numpy()
            )

    def get_energy(self, atoms):
        """
        Get the energy for the given atoms.
        """
        self.calculate(atoms, properties=["energy"])
        return self.results["energy"]

    def get_potential_energy(self, atoms):
        """
        Get the potential energy for the given atoms.
        """
        self.calculate(atoms, properties=["energy"])
        return self.results["energy"]

    def get_forces(self, atoms):
        """
        Get the forces for the given atoms.
        """
        self.calculate(atoms, properties=["forces"])
        return self.results["forces"]

    def get_hessian(self, atoms, hessian_method=None):
        """
        Get the Hessian matrix for the given atoms via autodiff (on the fly Hessian).
        """
        self.calculate(
            atoms,
            properties=["energy", "forces", "hessian"],
            hessian_method=hessian_method,
        )
        N = len(atoms)
        return self.results["hessian"].reshape(N * 3, N * 3)

    def get_hessian_autodiff(self, atoms):
        """
        Get the Hessian matrix for the given atoms via autodiff (on the fly Hessian).
        """
        self.calculate(
            atoms, properties=["energy", "forces", "hessian"], hessian_method="autodiff"
        )
        N = len(atoms)
        return self.results["hessian"].reshape(N * 3, N * 3)

    def get_hessian_prediction(self, atoms):
        """
        Get the Hessian matrix for the given atoms via prediction (from the model).
        """
        self.calculate(
            atoms, properties=["energy", "forces", "hessian"], hessian_method="predict"
        )
        N = len(atoms)
        return self.results["hessian"].reshape(N * 3, N * 3)

    def get_eigen_autodiff(self, atoms):
        """
        Get the eigenvalues and eigenvectors for the given atoms via autodiff (on the fly eigenvalues).
        """
        self.calculate(
            atoms,
            properties=["energy", "forces", "hessian", "eigen"],
            hessian_method="autodiff",
        )
        return self.results["eigenvalues"], self.results["eigenvectors"]


if __name__ == "__main__":
    import os
    from ase.vibrations import Vibrations

    # Create a simple water molecule for testing
    atoms = Atoms(
        symbols="H2O",
        positions=[
            [0.0, 0.0, 0.0],  # O
            [0.0, 0.757, 0.587],  # H
            [0.0, -0.757, 0.587],  # H
        ],
    )

    # Initialize calculator with default checkpoint path
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid checkpoint path")
        exit()

    calculator = EquiformerASECalculator(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        # hessian_method="predict"
    )

    # Attach calculator to atoms
    atoms.calc = calculator

    # Calculate energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"Energy: {energy:.6f} eV")
    print(f"Forces shape: {forces.shape}")
    print(f"Forces:\n{forces}")

    # To get hessian, we need to explicitly calculate it through the calculator
    calculator.calculate(atoms, properties=["energy", "forces", "hessian"])
    hessian = calculator.results["hessian"]
    print(f"Hessian shape: {hessian.shape}")

    eigenvalues, eigenvectors = calculator.get_eigen_autodiff(atoms)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors: {eigenvectors.shape}")

    # Compare with ASE's Vibrations class
    print("\n" + "=" * 40)
    print("Comparison with ASE's Vibrations class")
    vib = Vibrations(atoms)
    vib.run()
    vib.summary()
    vib.clean()
