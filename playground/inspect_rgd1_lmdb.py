import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdMolAlign import GetBestRMS
import io
import base64
from IPython.display import Image
import os
import py3Dmol
import webbrowser

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.equiformer_torch_calculator import EquiformerTorchCalculator
from gadff.align_unordered_mols import compute_rmsd
from gadff.plot_molecules import plot_molecule_mpl, plot_molecule_py3dmol

this_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(this_dir, "plots_gad")


def compute_molecular_linearity(coords, atomic_numbers, method="moment_ratio"):
    """
    Compute how "linear" a molecule is using different metrics.

    Args:
        coords: torch.Tensor of shape (N, 3) with atomic coordinates
        atomic_numbers: torch.Tensor of shape (N,) with atomic numbers
        method: str, method to use ("moment_ratio", "pca_ratio", "asphericity")

    Returns:
        float: Linearity measure. Higher values = more linear
               - moment_ratio: ratio of largest to smallest principal moment. higher = more linear
               - pca_ratio: ratio of largest to smallest PCA eigenvalue
               - asphericity: asphericity parameter
    """
    coords = coords.clone().float()
    atomic_numbers = atomic_numbers.clone().float()

    # Center coordinates at center of mass (weighted by atomic mass)
    # Using atomic number as mass approximation
    masses = atomic_numbers.unsqueeze(1)  # Shape: (N, 1)
    center_of_mass = (coords * masses).sum(dim=0) / masses.sum()
    coords_centered = coords - center_of_mass

    if method == "moment_ratio":
        # Compute moment of inertia tensor
        Idm = torch.zeros(3, 3)
        for i in range(len(coords_centered)):
            r = coords_centered[i]
            m = masses[i].item()
            # I_ij = m * (r^2 * delta_ij - r_i * r_j)
            r_squared = torch.sum(r**2)
            Idm += m * (r_squared * torch.eye(3) - torch.outer(r, r))

        # Get principal moments (eigenvalues)
        eigenvals = torch.linalg.eigvals(Idm).real
        eigenvals = torch.sort(eigenvals)[0]  # Sort ascending

        # Linearity ratio: largest / smallest moment
        # Add small epsilon to avoid division by zero
        linearity = eigenvals[-1] / (eigenvals[0] + 1e-10)
        return linearity.item()

    elif method == "pca_ratio":
        # PCA-based linearity
        # Weight coordinates by atomic masses
        coords_weighted = coords_centered * torch.sqrt(masses)
        cov_matrix = torch.cov(coords_weighted.T)
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        eigenvals = torch.sort(eigenvals)[0]

        # Ratio of largest to smallest eigenvalue
        linearity = eigenvals[-1] / (eigenvals[0] + 1e-10)
        return linearity.item()

    elif method == "asphericity":
        # Asphericity parameter: how much the molecule deviates from spherical
        coords_weighted = coords_centered * torch.sqrt(masses)
        cov_matrix = torch.cov(coords_weighted.T)
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        eigenvals = torch.sort(eigenvals)[0]

        # Asphericity = (lambda_max - (lambda_mid + lambda_min)/2) / (lambda_max + lambda_mid + lambda_min)
        lambda_max, lambda_mid, lambda_min = eigenvals[2], eigenvals[1], eigenvals[0]
        asphericity = (lambda_max - (lambda_mid + lambda_min) / 2) / (
            lambda_max + lambda_mid + lambda_min + 1e-10
        )
        return asphericity.item()

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_dataset_linearity(dataset, sample_idx=None, max_samples=100):
    """
    Analyze linearity across the dataset to find most/least linear molecules.

    Args:
        dataset: LmdbDataset
        max_samples: int, maximum number of samples to analyze

    Returns:
        list: List of (sample_idx, linearity_reactant, linearity_product, linearity_transition, natoms)
    """
    results = []

    if sample_idx is None:
        sample_idx = range(min(len(dataset), max_samples))

    for i in sample_idx:
        sample = dataset[i]
        natoms = len(sample.z)

        try:
            linearity_reactant = compute_molecular_linearity(
                sample.pos_reactant, sample.z, "moment_ratio"
            )
            linearity_product = compute_molecular_linearity(
                sample.pos_product, sample.z, "moment_ratio"
            )
            linearity_transition = compute_molecular_linearity(
                sample.pos_transition, sample.z, "moment_ratio"
            )

            results.append(
                (i, linearity_reactant, linearity_product, linearity_transition, natoms)
            )
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(plot_dir, exist_ok=True)

    # Load the LMDB dataset
    print("Loading RGD1 dataset")
    dataset = LmdbDataset("data/rgd1/rgd1_minimal_train.lmdb")
    print(f"Dataset size: {len(dataset)}")

    # Find smallest nsmallest samples (fewest atoms)
    nsmallest = 30
    smallest_samples = []  # List of (idx, natoms) tuples
    for i in range(len(dataset)):
        sample = dataset[i]
        natoms = len(sample.z)
        smallest_samples.append((i, natoms))

        # Keep only the 10 smallest samples
        if len(smallest_samples) > nsmallest:
            smallest_samples.sort(key=lambda x: x[1])
            smallest_samples = smallest_samples[:nsmallest]

    # Final sort to get the 10 smallest
    smallest_samples.sort(key=lambda x: x[1])
    smallest_samples = smallest_samples[:nsmallest]

    print(f"{nsmallest} smallest samples:")
    for rank, (idx, natoms) in enumerate(smallest_samples, 1):
        print(f"  {rank}. Sample {idx}: {natoms} atoms")

    # Use the smallest samples for plotting
    print(f"\nWill plot the first 5 smallest molecules...")

    # # Collect all the reactions with the same atoms
    # reactions_with_same_atoms = []
    # reactions_with_same_atoms_idx = []
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     if sample.element_counts_string == smallest_sample.element_counts_string:
    #         reactions_with_same_atoms.append(sample)
    #         reactions_with_same_atoms_idx.append(i)
    # print(f"Number of reactions with {smallest_sample.element_counts_string} atoms: {len(reactions_with_same_atoms)}")

    # Plot the first 5 smallest molecules and compute linearity
    for i in range(min(5, len(smallest_samples))):
        sample_idx, natoms = smallest_samples[i]
        sample = dataset[sample_idx]

        print(f"\nPlotting molecule {i+1}/5 (Sample {sample_idx}, {natoms} atoms)")

        # Compute linearity for each structure
        linearity_reactant = compute_molecular_linearity(
            sample.pos_reactant, sample.z, "moment_ratio"
        )
        linearity_product = compute_molecular_linearity(
            sample.pos_product, sample.z, "moment_ratio"
        )
        linearity_transition = compute_molecular_linearity(
            sample.pos_transition, sample.z, "moment_ratio"
        )

        asphericity_reactant = compute_molecular_linearity(
            sample.pos_reactant, sample.z, "asphericity"
        )
        asphericity_product = compute_molecular_linearity(
            sample.pos_product, sample.z, "asphericity"
        )
        asphericity_transition = compute_molecular_linearity(
            sample.pos_transition, sample.z, "asphericity"
        )

        print(f"  Linearity (moment ratio):")
        print(f"    Reactant:   {linearity_reactant:.2f}")
        print(f"    Product:    {linearity_product:.2f}")
        print(f"    Transition: {linearity_transition:.2f}")

        # Create unique titles and filenames for each sample
        title_suffix = f"{sample_idx}"

        plot_molecule_mpl(
            coords=sample.pos_reactant,
            title=f"Reactant {title_suffix}",
            plot_dir=plot_dir,
            atomic_numbers=sample.z,
            save=True,
        )
        plot_molecule_mpl(
            coords=sample.pos_product,
            title=f"Product {title_suffix}",
            plot_dir=plot_dir,
            atomic_numbers=sample.z,
            save=True,
        )
        plot_molecule_mpl(
            coords=sample.pos_transition,
            title=f"Transition {title_suffix}",
            plot_dir=plot_dir,
            atomic_numbers=sample.z,
            save=True,
        )

        # reactant_view = plot_molecule_py3dmol(
        #     coords=sample.pos_reactant,
        #     smiles=sample.smiles_reactant,
        #     title=f"Reactant py3Dmol {title_suffix}",
        #     atomic_numbers=sample.z,
        #     save_path=os.path.join(plot_dir, f"reactant_py3dmol_{title_suffix}"),
        #     save_png=True,
        #     save_html=False,
        # )

    # Print information about the smallest sample
    smallest_sample = dataset[smallest_samples[0][0]]
    print(
        f"\nSample information for smallest sample (Sample {smallest_samples[0][0]}):"
    )
    for k, v in smallest_sample.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)

    # Analyze linearity across a subset of the dataset
    smallest_samples_idx = [idx for idx, _ in smallest_samples]
    linearity_results = analyze_dataset_linearity(
        dataset, sample_idx=smallest_samples_idx
    )

    if linearity_results:
        # Sort by average linearity across all three structures
        linearity_results_avg = [
            (idx, (lr + lp + lt) / 3, lr, lp, lt, natoms)
            for idx, lr, lp, lt, natoms in linearity_results
        ]
        linearity_results_avg.sort(key=lambda x: x[1], reverse=True)

        print(f"\nMost linear molecules (top 5):")
        for i, (idx, avg_linearity, lr, lp, lt, natoms) in enumerate(
            linearity_results_avg[:5]
        ):
            print(
                f"  {i+1}. Sample {idx} ({natoms} atoms): avg={avg_linearity:.1f} (R={lr:.1f}, P={lp:.1f}, T={lt:.1f})"
            )

        print(f"\nLeast linear molecules (bottom 5):")
        for i, (idx, avg_linearity, lr, lp, lt, natoms) in enumerate(
            linearity_results_avg[-5:]
        ):
            print(
                f"  {i+1}. Sample {idx} ({natoms} atoms): avg={avg_linearity:.1f} (R={lr:.1f}, P={lp:.1f}, T={lt:.1f})"
            )

        # Also sort by maximum linearity change during reaction
        linearity_changes = [
            (idx, max(lr, lp, lt) - min(lr, lp, lt), lr, lp, lt, natoms)
            for idx, lr, lp, lt, natoms in linearity_results
        ]
        linearity_changes.sort(key=lambda x: x[1], reverse=True)

        print(f"\nLargest linearity changes during reaction (top 5):")
        for i, (idx, change, lr, lp, lt, natoms) in enumerate(linearity_changes[:5]):
            print(
                f"  {i+1}. Sample {idx} ({natoms} atoms): Δ={change:.1f} (R={lr:.1f}, P={lp:.1f}, T={lt:.1f})"
            )
