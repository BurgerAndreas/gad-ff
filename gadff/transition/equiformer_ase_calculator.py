"""
ASE Calculator wrapper for Equiformer model.
"""

from typing import Optional
import numpy as np
import torch
from torch_geometric.data import Data
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.constraints import FixAtoms

from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs

from gadff.horm.training_module import PotentialModule, compute_extra_props

# ocpmodels/common/relaxation/ase_utils.py
def batch_to_atoms(batch):
    n_systems = batch.natoms.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell
    energies = batch.y.tolist()

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects


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
    
    Might need to reimplement EquiformerCalculator based on:
    ocpmodels/common/relaxation/ase_utils.py

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
        
        # ocpmodels/common/relaxation/ase_utils.py
        self.a2g = AtomsToGraphs(
            max_neigh=self.model.potential.max_neighbors,
            radius=self.model.potential.cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )
    
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

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
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
        # data_object = self.a2g.convert(atoms)
        # batch = data_list_collater([data_object], otf_graph=True)

        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric(atoms)
        batch = batch.to(self.device)

        if properties is None:
            properties = []
        if "eigen" in properties:
            properties.append("hessian")

        # Prepare batch with extra properties
        batch = compute_extra_props(
            batch, pos_require_grad="hessian" in properties
        )

        # Run prediction
        with torch.enable_grad():
            energy, forces, eigenoutputs = self.model.potential.forward(batch, eigen=True)

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
        Get the Hessian matrix for the given atoms via autodiff (on the fly Hessian).
        """
        self.calculate(atoms, properties=["energy", "forces", "hessian"])
        return self.results["hessian"]

    def get_eigen_autodiff(self, atoms):
        """
        Get the eigenvalues and eigenvectors for the given atoms via autodiff (on the fly eigenvalues).
        """
        self.calculate(atoms, properties=["energy", "forces", "hessian", "eigen"])
        return self.results["eigenvalues"], self.results["eigenvectors"]

    # Andreas: really not sure if this is correct
    def get_vibrational_analysis(self, atoms, filter_threshold=1.0, preserve_imaginary=True):
        """
        Compute vibrational modes using mass-weighted Hessian.

        Args:
            atoms: ASE Atoms object
            filter_threshold: Threshold for filtering small frequencies (cm^-1). 
                            Set to 0.0 to keep all modes.
            preserve_imaginary: If True, preserves imaginary frequencies regardless of threshold.
                              Important for transition state analysis.

        Returns:
            dict: Dictionary containing vibrational analysis results:
                - frequencies: vibrational frequencies in cm^-1 (negative = imaginary)
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

        # Filter out translational and rotational modes
        # For transition state analysis, preserve imaginary frequencies (negative eigenvalues)
        if filter_threshold > 0.0:
            if preserve_imaginary:
                # Keep imaginary frequencies (negative) and real frequencies above threshold
                vibrational_mask = (frequencies_cm < 0) | (frequencies_cm > filter_threshold)
            else:
                # Traditional filtering by absolute value
                vibrational_mask = np.abs(frequencies_cm) > filter_threshold
        else:
            # Keep all modes
            vibrational_mask = np.ones(len(frequencies_cm), dtype=bool)
            
        print(f"Masked {np.sum(~vibrational_mask)} modes (filter_threshold={filter_threshold}, preserve_imaginary={preserve_imaginary})")

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

    def analyze_stationary_point(self, atoms):
        """
        Analyze whether the structure is a minimum, transition state, or higher-order saddle point.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            dict: Analysis results containing:
                - point_type: str ("minimum", "transition_state", "higher_order_saddle")
                - n_imaginary: int (number of imaginary frequencies)
                - frequencies: array of all frequencies
                - imaginary_frequencies: array of imaginary frequencies only
        """
        vib_results = self.get_vibrational_analysis(atoms, filter_threshold=1.0, preserve_imaginary=True)
        frequencies = vib_results["frequencies"]
        
        imaginary_freqs = frequencies[frequencies < 0]
        n_imaginary = len(imaginary_freqs)
        
        if n_imaginary == 0:
            point_type = "minimum"
        elif n_imaginary == 1:
            point_type = "transition_state"
        else:
            point_type = "higher_order_saddle"
            
        return {
            "point_type": point_type,
            "n_imaginary": n_imaginary,
            "frequencies": frequencies,
            "imaginary_frequencies": imaginary_freqs,
            "all_results": vib_results
        }


if __name__ == "__main__":
    from gadff.path_config import find_project_root
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
    root_dir = find_project_root()
    checkpoint_path = os.path.join(root_dir, "ckpt/eqv2.ckpt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid checkpoint path")
        exit()

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

    # Compute vibrational analysis for transition state analysis
    print("\n=== Vibrational Analysis (Transition State Compatible) ===")
    vib_results = calculator.get_vibrational_analysis(atoms, preserve_imaginary=True)

    print(f"Number of vibrational modes: {len(vib_results['frequencies'])}")
    print(f"Vibrational frequencies (cm^-1):")
    imaginary_count = 0
    for i, freq in enumerate(vib_results["frequencies"]):
        if freq < 0:
            print(f"  Mode {i+1}: {freq:.2f} (i)")
            imaginary_count += 1
        else:
            print(f"  Mode {i+1}: {freq:.2f}")
    
    print(f"\nNumber of imaginary frequencies: {imaginary_count}")
    if imaginary_count == 1:
        print("This appears to be a transition state (1 imaginary frequency)")
    elif imaginary_count == 0:
        print("This appears to be a minimum (0 imaginary frequencies)")
    else:
        print(f"This has {imaginary_count} imaginary frequencies")

    # Compare with ASE's Vibrations class
    print("\n" + "="*40)
    print("Comparison with ASE's Vibrations class")
    vib = Vibrations(atoms)
    vib.run()
    vib.summary()
    vib.clean()
