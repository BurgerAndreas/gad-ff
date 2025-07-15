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
from gadff.equiformer_calculator import EquiformerTorchCalculator
from gadff.align_unordered_mols import compute_rmsd
from gadff.plot_molecules import plot_molecule_mpl, plot_molecule_py3dmol

this_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(this_dir, "plots_gad")


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

    # Plot the first 5 smallest molecules
    for i in range(min(5, len(smallest_samples))):
        sample_idx, natoms = smallest_samples[i]
        sample = dataset[sample_idx]

        print(f"\nPlotting molecule {i+1}/5 (Sample {sample_idx}, {natoms} atoms)")

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
