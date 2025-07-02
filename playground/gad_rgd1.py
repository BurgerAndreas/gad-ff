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
from gadff.plot_molecules import plot_molecule_mpl, plot_traj_mpl

from ase import Atoms
from ase.io import read
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from sella import Sella, Constraints
from gadff.equiformer_ase_calculator import EquiformerASECalculator

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
    return TGData(z=rgd1_data.z, pos=pos, natoms=rgd1_data.natoms)


def integrate_dynamics(
    initial_pos,
    z,
    natoms,
    calc,
    true_pos,
    force_field="gad",
    title=None,
    max_steps=100,
    convergence_threshold=1e-4,
    dt=0.01,
    print_every=100,
):
    assert force_field in ["gad", "forces"], f"Unknown force field: {force_field}"

    device = calc.model.device

    # Compute RMSD between initial guess and transition state
    rmsd_initial = rmsd(initial_pos.numpy(), true_pos.numpy())
    print(f"RMSD between initial guess and true state: {rmsd_initial:.6f} Å")

    # Follow the vector field using autograd Hessian
    print(f"\nFollowing {force_field} vector field")

    # Create initial batch from interpolated guess
    current_pos = initial_pos.clone().requires_grad_(True).to(device)
    tg_data_current = TGData(z=z, pos=current_pos, natoms=natoms)

    # Storage for trajectory
    trajectory_pos = []
    trajectory_energy = []
    trajectory_forces_norm = []
    trajectory_forces = []

    print(f"Starting {force_field} integration")
    for step in range(max_steps):
        # Create batch for current position
        batch_current = Batch.from_data_list(
            [
                TGData(
                    z=z,
                    pos=current_pos.clone().detach().requires_grad_(True),
                    natoms=natoms,
                )
            ]
        ).to(device)

        if force_field == "gad":
            # Compute GAD vector field
            gad_vector, out = calc.predict_gad_with_hessian(batch_current)
            energy = out["energy"]
            forces = gad_vector.reshape(-1, 3)
        elif force_field == "forces":
            # Compute forces
            energy, forces, eigenpred = calc.predict(batch_current)
            forces = forces.reshape(-1, 3)

        # Store trajectory
        trajectory_pos.append(current_pos.detach().cpu().clone())
        trajectory_forces.append(forces.detach().cpu())
        trajectory_energy.append(energy.item())

        # Check for convergence (small vector magnitude)
        forces_norm = torch.norm(forces).item()
        trajectory_forces_norm.append(forces_norm)

        if forces_norm < convergence_threshold:
            print(f"Converged at step {step}, forces norm: {forces_norm:.6f}")
            break

        # terminate if force norm was small for 50 steps
        if np.all(np.array(trajectory_forces_norm[-50:]) < 2.0):
            print("Force norm was small for 50 steps, terminating")
            break

        # Euler integration step
        current_pos = current_pos.detach() + dt * forces

        if step % print_every == 0:
            print(
                f"Step {step}: VF norm = {forces_norm:.6f}, Energy = {energy.item():.6f}"
            )

    # After convergence, compute final RMSD and eigenvalues
    print("\nFinal analysis")

    # Plot trajectory
    plot_traj_mpl(
        trajectory_pos,
        title=f"{force_field} Integration - {title}",
        plot_dir=plot_dir,
        atomic_numbers=z,
        save=True,
        # forces_traj=trajectory_gad,
    )

    # optionally save xyz to visualize with Mol* / Protein Viewer

    # Final position
    final_pos = trajectory_pos[-1]

    # Compute RMSD between converged structure and true transition state
    rmsd_final = rmsd(final_pos.numpy(), true_pos.numpy())
    print(
        f"RMSD between converged structure and true transition state: {rmsd_final:.6f} Å"
    )
    print(f"Improvement: {rmsd_initial - rmsd_final:.6f} Å")

    # Compute Hessian eigenvalues at final structure
    batch_final = Batch.from_data_list(
        [TGData(z=z, pos=final_pos.requires_grad_(True), natoms=natoms)]
    )

    print("Computing Hessian eigenvalues at converged structure")
    energy_final, forces_final, hessian, eigenvalues, eigenvectors, eigenpred = (
        calc.predict_with_hessian(batch_final)
    )

    print(f"Final energy: {energy_final.item():.6f}")
    print(
        f"Two lowest Hessian eigenvalues: {eigenvalues[0].item():.6f}, {eigenvalues[1].item():.6f}"
    )

    # Plot convergence
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(trajectory_energy)
    plt.xlabel("Integration Step")
    plt.ylabel("Energy (eV)")
    plt.title(f"Energy during {force_field} Integration")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(trajectory_forces_norm)
    plt.xlabel("Integration Step")
    plt.ylabel("Vector Norm")
    plt.title(f"{force_field} Convergence")
    plt.grid(True)

    plt.tight_layout(pad=0.1)
    fig_name = f"{force_field}_convergence.png"
    plt.savefig(os.path.join(plot_dir, fig_name), dpi=150, bbox_inches="tight")
    print(f"Saved convergence plot to {os.path.join(plot_dir, fig_name)}")
    return trajectory_pos, trajectory_energy, trajectory_forces_norm, trajectory_forces


def run_sella(pos_initial_guess, z, natoms, true_pos, calc=None):
    # Create ASE Atoms object from the initial guess position
    positions_np = (
        pos_initial_guess.detach().cpu().numpy()
        if torch.is_tensor(pos_initial_guess)
        else pos_initial_guess.numpy()
    )
    atomic_numbers_np = z.detach().cpu().numpy() if torch.is_tensor(z) else z.numpy()
    mol_ase = Atoms(numbers=atomic_numbers_np, positions=positions_np)

    if calc is None:
        mol_ase.calc = EMT()
        calcname = "EMT"
    else:
        mol_ase.calc = calc
        calcname = calc.__class__.__name__
    print(f"\nStarting Sella {calcname} optimization to find transition state")

    rmsd_initial = rmsd(
        positions_np,
        (
            true_pos.detach().cpu().numpy()
            if torch.is_tensor(true_pos)
            else true_pos.numpy()
        ),
    )
    print(f"RMSD between initial guess and true transition state: {rmsd_initial:.6f} Å")

    # Set up a Sella Dynamics object with improved parameters for TS search
    dyn = Sella(
        mol_ase,
        # constraints=cons,
        # trajectory=os.path.join(logfolder, "cu_sella_ts.traj"),
        order=1,  # Explicitly search for first-order saddle point
        # eta=5e-5,  # Smaller finite difference step for higher accuracy
        # delta0=5e-3,  # Larger initial trust radius for TS search
        # gamma=0.1,  # Much tighter convergence for iterative diagonalization
        # rho_inc=1.05,  # More conservative trust radius adjustment
        # rho_dec=3.0,  # Allow larger trust radius changes
        # sigma_inc=1.3,  # Larger trust radius increases
        # sigma_dec=0.5,  # More aggressive trust radius decreases
        log_every_n=100,
    )

    # Run with much tighter convergence
    dyn.run(1e-5, 2000)  # Much stricter convergence criterion
    print("Sella optimization completed!")

    # Get the final structure, convert to np, compute RMSD to true transition state
    final_positions_sella = mol_ase.get_positions()

    # Compute RMSD between Sella-optimized structure and true transition state
    true_ts_positions = (
        true_pos.detach().cpu().numpy()
        if torch.is_tensor(true_pos)
        else true_pos.numpy()
    )
    rmsd_sella = rmsd(final_positions_sella, true_ts_positions)
    print(
        f"RMSD between Sella-optimized structure and true transition state: {rmsd_sella:.6f} Å"
    )
    print(f"Improvement: {rmsd_initial - rmsd_sella:.6f} Å")
    return mol_ase


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
    sample = dataset[104_000]
    print(sample)
    print(f"First sample keys: {sample.keys()}")
    print(f"Number of atoms: {sample.natoms}")
    print(f"Elements (z): {sample.z}")
    print(f"Reactant SMILES: {sample.smiles_reactant}")
    print(f"Product SMILES: {sample.smiles_product}")

    # Convert to torch_geometric format for transition state
    print("\nConverting to torch_geometric format")
    tg_data = convert_rgd1_to_tg_format(sample, state="transition")

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

    # Plot the reactant and transition state
    print("\nPlotting molecular structures")
    plot_molecule_mpl(
        sample.pos_reactant,
        atomic_numbers=sample.z,
        title="Reactant",
        plot_dir=plot_dir,
        save=True,
    )
    plot_molecule_mpl(
        sample.pos_transition,
        atomic_numbers=sample.z,
        title="Transition state",
        plot_dir=plot_dir,
        save=True,
    )
    plot_molecule_mpl(
        sample.pos_product,
        atomic_numbers=sample.z,
        title="Product",
        plot_dir=plot_dir,
        save=True,
    )

    ###################################################################################
    # Use linear interpolation as initial guess (simpler than geodesic for now)
    print("\nCreating initial guess using linear interpolation")

    # Get reactant and transition state positions
    pos_reactant = sample.pos_reactant
    pos_transition = sample.pos_transition
    pos_product = sample.pos_product

    alpha = 0.5  # midpoint
    # Linear interpolation between reactant and product
    # pos_initial_guess = (1 - alpha) * pos_reactant + alpha * pos_product
    pos_initial_guess_rts = (1 - alpha) * pos_reactant + alpha * pos_transition
    plot_molecule_mpl(
        pos_initial_guess_rts,
        atomic_numbers=sample.z,
        title="Initial guess R-TS interpolation",
        plot_dir=plot_dir,
        save=True,
    )

    pos_initial_guess_rp = (1 - alpha) * pos_reactant + alpha * pos_product
    plot_molecule_mpl(
        pos_initial_guess_rp,
        atomic_numbers=sample.z,
        title="Initial guess R-P interpolation",
        plot_dir=plot_dir,
        save=True,
    )

    ###################################################################################
    # Follow the GAD vector field to find the transition state
    print("=" * 60)
    print("Following GAD vector field to find transition state")
    traj, _, _, _ = integrate_dynamics(
        pos_reactant,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=1_000,
        dt=0.01,
        title="TS from R",
    )
    plot_molecule_mpl(
        traj[-1],
        atomic_numbers=sample.z,
        title="GAD Optimized TS from R",
        plot_dir=plot_dir,
        save=True,
    )

    # Follow the GAD vector field to find the transition state
    traj, _, _, _ = integrate_dynamics(
        pos_initial_guess_rp,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=1_000,
        dt=0.01,
        title="TS from R-P interpolation",
    )
    plot_molecule_mpl(
        traj[-1],
        atomic_numbers=sample.z,
        title="GAD Optimized TS from R-P interpolation",
        plot_dir=plot_dir,
        save=True,
    )

    ###################################################################################
    print("=" * 60)
    print("Following Sella to find transition state")

    # See if Sella can find the transition state
    asecalc = EquiformerASECalculator(device=device)

    # Linear interpolation between reactant and product
    mol_ase = run_sella(
        pos_initial_guess_rp, sample.z, sample.natoms, sample.pos_transition, asecalc
    )
    plot_molecule_mpl(
        mol_ase.get_positions(),
        atomic_numbers=sample.z,
        title="Sella Equiformer Optimized TS from R-P interpolation",
        plot_dir=plot_dir,
        save=True,
    )

    # Linear interpolation between reactant and TS
    mol_ase = run_sella(
        pos_initial_guess_rts, sample.z, sample.natoms, sample.pos_transition, asecalc
    )
    plot_molecule_mpl(
        mol_ase.get_positions(),
        atomic_numbers=sample.z,
        title="Sella Equiformer Optimized TS from R-TS interpolation",
        plot_dir=plot_dir,
        save=True,
    )

    # Start from reactant
    mol_ase = run_sella(
        pos_reactant, sample.z, sample.natoms, sample.pos_transition, asecalc
    )
    plot_molecule_mpl(
        mol_ase.get_positions(),
        atomic_numbers=sample.z,
        title="Sella Equiformer Optimized TS from reactant",
        plot_dir=plot_dir,
        save=True,
    )

    ###################################################################################
    # # See if Sella can find the transition state
    # mol_ase = run_sella(
    #     pos_initial_guess, sample.z, sample.natoms, sample.pos_transition
    # )

    # # Plot the Sella-optimized structure
    # plot_molecule_mpl(
    #     mol_ase.get_positions(),
    #     atomic_numbers=sample.z,
    #     title="Sella EMT Optimized TS",
    #     plot_dir=plot_dir,
    #     save=True,
    # )

    ###################################################################################
    # Follow the forces to find the reactant minimum
    print("=" * 60)
    print("\nFollowing forces to find reactant minimum")

    # start by interpolating true transition state and reactant
    alpha = 0.5
    pos_initial_guess = (1 - alpha) * pos_reactant + alpha * pos_transition

    # Follow the forces to find the reactant minimum
    trajectory_pos, _, _, _ = integrate_dynamics(
        pos_initial_guess,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_reactant,
        force_field="forces",
        dt=0.01,  # time step
        max_steps=100,  # maximum optimization steps
        convergence_threshold=1e-4,  # convergence criterion for forces
        title="R-TS interpolation",
    )
    plot_molecule_mpl(
        trajectory_pos[-1],
        atomic_numbers=sample.z,
        title="Optimized Minimum from R-TS interpolation",
        plot_dir=plot_dir,
        save=True,
    )


if __name__ == "__main__":
    main()
