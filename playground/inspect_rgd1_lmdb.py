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
from gadff.equiformer_calculator import EquiformerCalculator
from gadff.align_unordered_mols import rmsd
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

    # # Find smallest sample (fewest atoms)
    # smallest_sample_idx = 0
    # smallest_sample_natoms = 1000000000
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     natoms = len(sample.z)
    #     if natoms < smallest_sample_natoms:
    #         smallest_sample_idx = i
    #         smallest_sample_natoms = natoms
    # print(f"Smallest sample: {smallest_sample_idx} with {smallest_sample_natoms} atoms")

    smallest_sample_idx = 104_000
    smallest_sample = dataset[smallest_sample_idx]

    # # Collect all the reactions with the same atoms
    # reactions_with_same_atoms = []
    # reactions_with_same_atoms_idx = []
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     if sample.element_counts_string == smallest_sample.element_counts_string:
    #         reactions_with_same_atoms.append(sample)
    #         reactions_with_same_atoms_idx.append(i)
    # print(f"Number of reactions with {smallest_sample.element_counts_string} atoms: {len(reactions_with_same_atoms)}")

    # Plot the reactions
    plot_molecule_mpl(
        smallest_sample.pos_reactant,
        "Reactant",
        plot_dir,
        atomic_numbers=smallest_sample.z,
    )
    plot_molecule_mpl(
        smallest_sample.pos_product,
        "Product",
        plot_dir,
        atomic_numbers=smallest_sample.z,
    )
    plot_molecule_mpl(
        smallest_sample.pos_transition,
        "Transition",
        plot_dir,
        atomic_numbers=smallest_sample.z,
    )

    reactant_view = plot_molecule_py3dmol(
        smallest_sample.pos_reactant,
        smallest_sample.smiles_reactant,
        atomic_numbers=smallest_sample.z,
        title="Reactant (py3Dmol)",
        save_path=os.path.join(plot_dir, "reactant_py3dmol"),
        save_png=True,
        save_html=False,
    )

    print("\nSample:")
    for k, v in smallest_sample.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
