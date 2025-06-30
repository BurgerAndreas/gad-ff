"""
ASE Calculator wrapper for Equiformer model.
"""

from typing import Optional
import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
from torch_geometric.data import Data

from gadff.horm.training_module import PotentialModule, compute_extra_props


"""
Might need to reimplement based on this:
ocpmodels/common/relaxation/ase_utils.py
ocpmodels/preprocessing/atoms_to_graphs.py
"""


def ase_atoms_to_torch_geometric(atoms):
    """
    Convert ASE Atoms object to torch_geometric Data format expected by Equiformer.

    Args:
        atoms: ASE Atoms object

    Returns:
        Data: torch_geometric Data object with required attributes
    """
    positions = atoms.get_positions().astype(np.float32)
    atomic_nums = atoms.get_atomic_numbers()

    # Create one-hot encoding for supported elements (H, C, N, O)
    # This matches the format used in the training data
    element_mapping = {1: 0, 6: 1, 7: 2, 8: 3}  # H, C, N, O
    one_hot_matrix = np.zeros((len(atomic_nums), 4), dtype=np.int64)

    for i, atom_num in enumerate(atomic_nums):
        if atom_num in element_mapping:
            one_hot_matrix[i, element_mapping[atom_num]] = 1
        else:
            raise ValueError(f"Unsupported element with atomic number {atom_num}")

    # Create batch indices (all atoms belong to the same molecule)
    batch_indices = np.zeros(len(atomic_nums), dtype=np.int64)

    # Convert to torch tensors
    data = Data(
        pos=torch.tensor(positions, dtype=torch.float32),
        one_hot=torch.tensor(one_hot_matrix, dtype=torch.int64),
        charges=torch.tensor(atomic_nums, dtype=torch.int64),
        batch=torch.tensor(batch_indices, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        # Add dummy energy for compatibility (will be overwritten by prediction)
        energy=torch.tensor(0.0, dtype=torch.float32),
    )

    return data


class EquiformerCalculator(Calculator):
    """
    Equiformer ASE Calculator.

    Args:
        checkpoint_path: Path to the Equiformer model checkpoint
        device: Optional device specification (defaults to auto-detect)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
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
        self.results = {}

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"Loading Equiformer model from: {checkpoint_path}")
        self.model = PotentialModule.load_from_checkpoint(
            checkpoint_path, strict=False, map_location=self.device
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set implemented properties
        self.implemented_properties = ["energy", "forces", "hessian"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties for the given atoms.

        Args:
            atoms: ASE Atoms object
            properties: List of properties to compute (used by ASE internally)
            system_changes: System changes since last calculation (used by ASE internally)
        """
        # Call base class to set atoms attribute
        Calculator.calculate(self, atoms)

        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric(atoms)
        batch = batch.to(self.device)

        if "eigen" in properties:
            properties.append("hessian")

        # Prepare batch with extra properties
        batch = compute_extra_props(batch, pos_require_grad="hessian" in properties)

        # Run prediction
        with torch.enable_grad():
            energy, forces = self.model.potential.forward(batch)

        # Store results
        self.results = {}

        # Energy is per molecule, extract scalar value
        self.results["energy"] = float(energy.detach().cpu().item())

        # Forces shape: [n_atoms, 3]
        self.results["forces"] = forces.detach().cpu().numpy()

        if "hessian" in properties:
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
            self.results["hessian"] = hessian.detach().cpu().numpy()

        if "eigen" in properties:
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            smallest_eigenvals = eigenvalues[:2]
            smallest_eigenvecs = eigenvectors[:, :2]
            self.results["eigenvalues"] = smallest_eigenvals.detach().cpu().numpy()
            self.results["eigenvectors"] = (
                smallest_eigenvecs.T.view(2, N, 3).detach().cpu().numpy()
            )

    def get_hessian_autodiff(self, atoms):
        """
        Get the Hessian matrix for the given atoms.
        """
        self.calculate(atoms, properties=["energy", "forces", "hessian"])
        return self.results["hessian"]

    def get_eigen_autodiff(self, atoms):
        """
        Get the eigenvalues and eigenvectors for the given atoms.
        """
        self.calculate(atoms, properties=["energy", "forces", "hessian", "eigen"])
        return self.results["eigenvalues"], self.results["eigenvectors"]

    def get_vibrational_analysis(self, atoms):
        """
        Compute vibrational modes using mass-weighted Hessian.

        Args:
            atoms: ASE Atoms object

        Returns:
            dict: Dictionary containing vibrational analysis results:
                - frequencies: vibrational frequencies in cm^-1
                - normal_modes: normal mode eigenvectors (mass-weighted)
                - reduced_masses: reduced masses for each mode
                - force_constants: force constants for each mode
        """
        # Get the Hessian matrix
        self.calculate(atoms, properties=["energy", "forces", "hessian"])
        hessian = self.results["hessian"]

        # Get atomic masses in atomic units
        masses = atoms.get_masses()  # in amu
        masses_au = masses * 1822.888486  # Convert amu to atomic units

        # Create mass matrix (diagonal matrix with masses repeated 3 times for x,y,z)
        N = len(atoms)
        mass_matrix = np.zeros((3 * N, 3 * N))
        for i in range(N):
            mass_matrix[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = masses_au[i] * np.eye(3)

        # Create mass-weighted Hessian: H_mw = M^(-1/2) * H * M^(-1/2)
        mass_matrix_sqrt = np.sqrt(mass_matrix)
        mass_matrix_inv_sqrt = np.linalg.inv(mass_matrix_sqrt)

        # Mass-weighted Hessian
        hessian_mw = mass_matrix_inv_sqrt @ hessian @ mass_matrix_inv_sqrt

        # Solve eigenvalue problem for mass-weighted Hessian
        eigenvalues, eigenvectors = np.linalg.eigh(hessian_mw)

        # Convert eigenvalues to frequencies
        # ω = sqrt(λ) where λ are eigenvalues of mass-weighted Hessian
        # Convert from atomic units to cm^-1
        # 1 Hartree = 219474.631 cm^-1
        # 1 atomic unit of frequency = sqrt(Hartree/amu) = 5140.487 cm^-1
        frequencies_cm = np.sqrt(np.abs(eigenvalues)) * 5140.487

        # Handle negative eigenvalues (imaginary frequencies)
        frequencies_cm = np.where(eigenvalues < 0, -frequencies_cm, frequencies_cm)

        # Convert eigenvectors back to Cartesian coordinates
        # Normal modes in Cartesian coordinates: Q = M^(-1/2) * q
        normal_modes_cart = mass_matrix_inv_sqrt @ eigenvectors

        # Calculate reduced masses for each mode
        # μ = 1 / (q^T * q) where q are mass-weighted normal modes
        reduced_masses = 1.0 / np.sum(eigenvectors**2, axis=0)

        # Calculate force constants
        # k = μ * ω^2
        force_constants = reduced_masses * (frequencies_cm / 5140.487) ** 2

        # Filter out translational and rotational modes (first 6 modes)
        # These typically have very small frequencies
        vibrational_mask = (
            np.abs(frequencies_cm) > 1.0
        )  # Filter out modes with |freq| < 1 cm^-1
        print(f"Masked {np.sum(~vibrational_mask)} modes")

        return {
            "frequencies": frequencies_cm[vibrational_mask],
            "normal_modes": normal_modes_cart[:, vibrational_mask],
            "reduced_masses": reduced_masses[vibrational_mask],
            "force_constants": force_constants[vibrational_mask],
            "all_frequencies": frequencies_cm,
            "all_normal_modes": normal_modes_cart,
            "all_reduced_masses": reduced_masses,
            "all_force_constants": force_constants,
            "vibrational_mask": vibrational_mask,
        }


def example_usage():
    """
    Example usage of the EquiformerCalculator.
    """
    from ase import Atoms
    from gadff.path_config import find_project_root
    import os

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
    root_dir = find_project_root()
    checkpoint_path = os.path.join(root_dir, "ckpt/eqv2.ckpt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid checkpoint path")
        return

    calculator = EquiformerCalculator(checkpoint_path=checkpoint_path)

    # Attach calculator to atoms
    # atoms.set_calculator(calculator)
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

    # Compute vibrational analysis
    print("\n=== Vibrational Analysis ===")
    vib_results = calculator.get_vibrational_analysis(atoms)

    print(f"Number of vibrational modes: {len(vib_results['frequencies'])}")
    print(f"Vibrational frequencies (cm^-1):")
    for i, freq in enumerate(vib_results["frequencies"]):
        print(f"  Mode {i+1}: {freq:.2f}")

    print(f"\nNormal modes shape: {vib_results['normal_modes'].shape}")
    print(f"Reduced masses: {vib_results['reduced_masses']}")
    print(f"Force constants: {vib_results['force_constants']}")

    # Show all frequencies including translational/rotational modes
    print(f"\nAll frequencies (including translational/rotational):")
    for i, freq in enumerate(vib_results["all_frequencies"]):
        mode_type = "vibrational" if vib_results["vibrational_mask"][i] else "trans/rot"
        print(f"  Mode {i+1}: {freq:.2f} cm^-1 ({mode_type})")


if __name__ == "__main__":
    example_usage()
