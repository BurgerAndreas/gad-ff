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

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.equiformer_calculator import EquiformerCalculator
from gadff.align_unordered_mols import rmsd
from gadff.plot_molecules import plot_molecule_from_3d

this_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(this_dir, "plots_gad")

def convert_rgd1_to_tg_format(rgd1_data, state="transition"):
    """Convert RGD1 data format to standard torch_geometric format"""
    
    # Choose which positions to use
    if state == "transition":
        pos = rgd1_data.pos_transition
    elif state == "reactant":
        pos = rgd1_data.pos_reactant
    elif state == "product":
        pos = rgd1_data.pos_product
    else:
        raise ValueError(f"Unknown state: {state}")
    
    # Create standard torch_geometric Data object
    return TGData(
        z=rgd1_data.z,
        pos=pos, 
        natoms=rgd1_data.natoms
    )
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load the LMDB dataset
    print("Loading RGD1 dataset")
    dataset = LmdbDataset("data/rgd1/rgd1_minimal_train.lmdb")
    print(f"Dataset size: {len(dataset)}")
    
    # Get the first sample
    print("\nLoading first sample")
    # Smallest sample: 104_000 with 4 atoms
    first_sample = dataset[104_000]
    print(first_sample)
    print(f"First sample keys: {first_sample.keys()}")
    print(f"Number of atoms: {first_sample.natoms}")
    print(f"Elements (z): {first_sample.z}")
    print(f"Reactant SMILES: {first_sample.smiles_reactant}")
    print(f"Product SMILES: {first_sample.smiles_product}")
    
    # Convert to torch_geometric format for transition state
    print("\nConverting to torch_geometric format")
    tg_data = convert_rgd1_to_tg_format(first_sample, state="transition")
    
    # Create batch (equiformer expects batch format)
    batch = Batch.from_data_list([tg_data])
    print(f"Batch shape - pos: {batch.pos.shape}, z: {batch.z.shape}")
    
    # Initialize equiformer calculator
    print("\nInitializing EquiformerCalculator")
    calc = EquiformerCalculator(device=device)
    
    # Run prediction
    print("\nRunning prediction")
    energy, forces, eigenpred = calc.predict(batch)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Energy: {energy.item():.6f}")
    print(f"Forces shape: {forces.shape}")
    print(f"Forces norm: {torch.norm(forces).item():.6f}")
    print(f"Eigenprediction keys: {list(eigenpred.keys())}")
    
    ###################################################################################
    # Follow the GAD vector field to find the transition state
    
    # Plot the reactant and transition state using rdkit
    print("\nPlotting molecular structures")
    
    plot_molecule_from_3d(first_sample.pos_reactant, "Reactant", plot_dir)
    plot_molecule_from_3d(first_sample.pos_product, "Product", plot_dir)
    
    # Use linear interpolation as initial guess (simpler than geodesic for now)
    print("\nCreating initial guess using linear interpolation")
    
    # Get reactant and transition state positions
    pos_reactant = first_sample.pos_reactant
    pos_transition = first_sample.pos_transition
    
    # Linear interpolation between reactant and transition state
    alpha = 0.5  # midpoint
    pos_initial_guess = (1 - alpha) * pos_reactant + alpha * pos_transition
    
    # Compute RMSD between initial guess and transition state
    rmsd_initial = rmsd(pos_initial_guess.numpy(), pos_transition.numpy())
    print(f"RMSD between initial guess and transition state: {rmsd_initial:.6f} Å")
    
    # Follow the GAD vector field using autograd Hessian
    print("\nFollowing GAD vector field")
    
    # Create initial batch from interpolated guess
    current_pos = pos_initial_guess.clone().requires_grad_(True).to(device)
    tg_data_current = TGData(z=first_sample.z, pos=current_pos, natoms=first_sample.natoms)
    
    # Integration parameters
    dt = 0.01  # time step
    max_steps = 1000  # maximum integration steps
    convergence_threshold = 1e-4  # convergence criterion
    
    # Storage for trajectory
    trajectory_pos = [current_pos.detach().clone()]
    trajectory_energy = []
    trajectory_forces_norm = []
    
    print("Starting GAD integration")
    for step in range(max_steps):
        # Create batch for current position
        batch_current = Batch.from_data_list([TGData(
            z=first_sample.z, 
            pos=current_pos.clone().detach().requires_grad_(True),
            natoms=first_sample.natoms
        )]).to(device)
        
        # Compute GAD vector field
        gad_vector, out = calc.predict_gad_with_hessian(batch_current)
        gad_vector = gad_vector.reshape(-1, 3)
        
        # Check for convergence (small GAD vector magnitude)
        gad_norm = torch.norm(gad_vector).item()
        trajectory_forces_norm.append(gad_norm)
        
        if gad_norm < convergence_threshold:
            print(f"Converged at step {step}, GAD norm: {gad_norm:.6f}")
            break
        
        # Euler integration step
        current_pos = current_pos.detach() + dt * gad_vector
        current_pos.requires_grad_(True)
        
        # Store trajectory
        trajectory_pos.append(current_pos.detach().cpu().clone())
        
        # Compute energy for monitoring
        energy = out["energy"]
        trajectory_energy.append(energy.item())
        
        if step % 10 == 0:
            print(f"Step {step}: GAD norm = {gad_norm:.6f}, Energy = {energy.item():.6f}")
    
    # After convergence, compute final RMSD and eigenvalues
    print("\nFinal analysis")
    
    # Final position
    final_pos = trajectory_pos[-1]
    
    # Compute RMSD between converged structure and true transition state
    rmsd_final = rmsd(final_pos.numpy(), pos_transition.numpy())
    print(f"RMSD between converged structure and true transition state: {rmsd_final:.6f} Å")
    print(f"Improvement: {rmsd_initial - rmsd_final:.6f} Å")
    
    # Compute Hessian eigenvalues at final structure
    batch_final = Batch.from_data_list([TGData(
        z=first_sample.z,
        pos=final_pos.requires_grad_(True),
        natoms=first_sample.natoms
    )])
    
    print("Computing Hessian eigenvalues at converged structure")
    energy_final, forces_final, hessian, eigenvalues, eigenvectors, eigenpred = calc.predict_with_hessian(batch_final)
    
    print(f"Final energy: {energy_final.item():.6f}")
    print(f"Two lowest Hessian eigenvalues: {eigenvalues[0].item():.6f}, {eigenvalues[1].item():.6f}")
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trajectory_energy)
    plt.xlabel('Integration Step')
    plt.ylabel('Energy (eV)')
    plt.title('Energy during GAD Integration')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(trajectory_forces_norm)
    plt.xlabel('Integration Step')
    plt.ylabel('GAD Vector Norm')
    plt.title('GAD Convergence')
    plt.grid(True)
    
    plt.tight_layout()
    fig_name = f"gad_convergence_{first_sample.smiles_reactant}_{first_sample.smiles_product}.png"
    plt.savefig(os.path.join(plot_dir, fig_name), dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot to {os.path.join(plot_dir, fig_name)}")

if __name__ == "__main__":
    main()
