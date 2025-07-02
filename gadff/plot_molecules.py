import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDetermineBonds
# from rdkit.Chem.Draw.IPythonConsole import drawMol3D  # draws a single molecule in 3D using py3Dmol
from rdkit.Chem.rdmolops import GetFormalCharge


import io
import base64
from IPython.display import Image
import os
import py3Dmol
import webbrowser
import plotly.io as pio
import seaborn as sns
# pio.renderers.default = "browser"

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.equiformer_calculator import EquiformerCalculator
from gadff.align_unordered_mols import rmsd
from ocpmodels.units import ELEMENT_TO_ATOMIC_NUMBER, ATOMIC_NUMBER_TO_ELEMENT

def plot_molecule_from_3d(pos, title, plot_dir, atomic_numbers, bonds=None):
    """
    Plot a 3D molecule from atomic coordinates.
    
    Parameters:
    -----------
    pos : torch.Tensor or np.ndarray
        Atomic positions with shape (n_atoms, 3)
    title : str
        Title for the plot and filename
    plot_dir : str
        Directory to save the plot
    atomic_numbers : list or np.ndarray, optional
        Atomic numbers for each atom (for coloring)
    bonds : list of tuples, optional
        List of (i, j) tuples representing bonds between atoms
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(pos):
        pos_np = pos.detach().cpu().numpy()
    else:
        pos_np = np.array(pos)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define atomic colors using seaborn pastel palette
    pastel_palette = sns.color_palette("pastel", 10)
    atomic_colors = {
        1: pastel_palette[0],   # H - light blue
        6: pastel_palette[1],   # C - light orange  
        7: pastel_palette[2],   # N - light green
        8: pastel_palette[3],   # O - light red
        9: pastel_palette[4],   # F - light purple
        15: pastel_palette[5],  # P - light brown
        16: pastel_palette[6],  # S - light pink
        17: pastel_palette[7],  # Cl - light gray
        35: pastel_palette[8],  # Br - light olive
        53: pastel_palette[9]   # I - light cyan
    }
    
    # Set colors and sizes based on atomic numbers
    if torch.is_tensor(atomic_numbers):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()
    colors = [atomic_colors.get(int(z), pastel_palette[9]) for z in atomic_numbers]
    # Bigger circles - size based on atomic number (larger for heavier atoms)
    sizes = [max(300, int(z) * 30) for z in atomic_numbers]
    
    # Plot atoms
    ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], 
               c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add bonds if provided
    if bonds is not None:
        for i, j in bonds:
            ax.plot([pos_np[i, 0], pos_np[j, 0]], 
                   [pos_np[i, 1], pos_np[j, 1]], 
                   [pos_np[i, 2], pos_np[j, 2]], 
                   'k-', alpha=0.6, linewidth=1.5)
    else:
        # Infer bonds using RDKit if atomic numbers are provided
        try:
            # Create RDKit molecule from atomic coordinates and numbers
            mol = Chem.RWMol()
            
            # Add atoms to molecule
            for i, atomic_num in enumerate(atomic_numbers):
                atom = Chem.Atom(int(atomic_num))
                mol.AddAtom(atom)
            
            # Add conformer with 3D coordinates
            conf = Chem.Conformer(len(pos_np))
            for i, (x, y, z) in enumerate(pos_np):
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))
            mol.AddConformer(conf)
            
            # Use RDKit's bond perception
            rdDetermineBonds.DetermineBonds(mol, charge=0)
            
            # Extract bonds and plot them
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                ax.plot([pos_np[i, 0], pos_np[j, 0]], 
                        [pos_np[i, 1], pos_np[j, 1]], 
                        [pos_np[i, 2], pos_np[j, 2]], 
                        'k-', alpha=0.6, linewidth=1.5)
                        
        except Exception as e:
            print(f"Warning: RDKit bond inference failed ({e}), falling back to distance-based method")
            # Fallback to distance-based method
            n_atoms = len(pos_np)
            bond_threshold = 2.0  # Angstroms
            
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dist = np.linalg.norm(pos_np[i] - pos_np[j])
                    if dist < bond_threshold:
                        ax.plot([pos_np[i, 0], pos_np[j, 0]], 
                                [pos_np[i, 1], pos_np[j, 1]], 
                                [pos_np[i, 2], pos_np[j, 2]], 
                                'k-', alpha=0.4, linewidth=1.0)
    
    # Add atom labels
    for i, (x, y, z) in enumerate(pos_np):
        if atomic_numbers is not None:
            element_symbol = ATOMIC_NUMBER_TO_ELEMENT[int(atomic_numbers[i])]
            ax.text(x, y, z, f'{element_symbol}{i}', fontsize=8, 
                   ha='center', va='center')
        else:
            ax.text(x, y, z, f'{i}', fontsize=8, 
                   ha='center', va='center')
    
    # Set labels and title
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(f'{title} - 3D Structure', fontsize=14, fontweight='bold')
    
    # Make axes equal
    max_range = np.array([pos_np[:, 0].max() - pos_np[:, 0].min(),
                         pos_np[:, 1].max() - pos_np[:, 1].min(),
                         pos_np[:, 2].max() - pos_np[:, 2].min()]).max() / 2.0
    
    mid_x = (pos_np[:, 0].max() + pos_np[:, 0].min()) * 0.5
    mid_y = (pos_np[:, 1].max() + pos_np[:, 1].min()) * 0.5
    mid_z = (pos_np[:, 2].max() + pos_np[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Improve the view
    ax.view_init(elev=30, azim=45)
    
    # Save the plot
    filename = f"{title.lower().replace(' ', '_').replace('-', '_')}_3d.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Saved 3D molecular structure to {filepath}")


def plot_molecule_py3dmol(coords, smiles: str, title: str = "Molecule", save_path: str = None, save_png: bool = False, save_html: bool = False, atomic_numbers=None):
    """Render an interactive 3-D view of a molecule using py3Dmol.

    Parameters
    ----------
    coords : torch.Tensor | np.ndarray
        Cartesian coordinates with shape (N, 3) corresponding to the atom
        order in *smiles*.
    smiles : str
        SMILES string describing the molecular graph.
    title : str, optional
        Title that will appear in the viewer.
    atomic_numbers : torch.Tensor | np.ndarray, optional
        Atomic numbers for each atom. If provided, will be used instead of
        trying to parse from SMILES.

    Returns
    -------
    py3Dmol.view
        A configured viewer instance that can be shown inside a notebook or
        whose HTML representation can be written to disk.
    """

    # Ensure NumPy array input
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    
    if atomic_numbers is not None and isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()

    # Map atomic numbers to element symbols
    atomic_number_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
    
    # Create XYZ block
    xyz_lines = [str(len(coords)), title]
    
    for i, (x, y, z) in enumerate(coords):
        atomic_num = int(atomic_numbers[i])
        symbol = atomic_number_to_symbol.get(atomic_num, f'X{atomic_num}')
        xyz_lines.append(f"{symbol} {float(x):.6f} {float(y):.6f} {float(z):.6f}")
    
    xyz_block = '\n'.join(xyz_lines)
    
    # Configure viewer with XYZ format
    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz_block, "xyz")
    view.setStyle({"stick": {}})
    view.setTitle(title)
    view.zoomTo()
        
    # # Build RDKit molecule (no additional hydrogens – assume coords already
    # # contain all atoms that appear in the SMILES string).
    # mol = Chem.MolFromSmiles(smiles)
    # # Original approach when atom counts match
    # conf = Chem.Conformer(mol.GetNumAtoms())

    # for idx, (x, y, z) in enumerate(coords):
    #     conf.SetAtomPosition(idx, (float(x), float(y), float(z)))

    # mol.AddConformer(conf, assignId=True)

    # # Convert to PDB (3-D) format understood by 3Dmol.js
    # pdb_block = Chem.MolToPDBBlock(mol)

    # # Configure viewer
    # view = py3Dmol.view(width=400, height=400)
    # view.addModel(pdb_block, "pdb")
    # view.setStyle({"stick": {}})
    # view.setTitle(title)
    # view.zoomTo()

    if save_path is not None:
        if save_html:
            # Write an HTML file so it can be opened in any browser
            save_path = save_path + ".html"
            with open(save_path, "w") as fh:
                fh.write(view._make_html())
            print(f"Saved interactive py3Dmol viewer to {save_path}")
            # Open the HTML file in the default browser
            # webbrowser.open(save_path)
        
        if save_png:
            # save as png
            fname = save_path + ".png"
            view.saveImage(fname)
            print(f"Saved screenshot to {fname}")
            
    return view