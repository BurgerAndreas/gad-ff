"""
GAD-RGD1: Gentlest Ascent Dynamics for Transition State Finding

This script implements Gentlest Ascent Dynamics (GAD) to find transition states
using different eigenvalue calculation methods for the Hessian matrix.

Available eigen methods:
- "qr": QR-based projector method (default)
- "svd": SVD-based projector method
- "inertia": Inertia tensor-based projector with auto-linearity detection
- "geo": Use Geometric library (external dependency)
- "ase": Use ASE library (external dependency)
- "eckart": Eckart frame alignment with principal axes

Example commands:

# Test all eigen methods
python playground/gad_rgd1.py --eigen-method qr
python playground/gad_rgd1.py --eigen-method svd
python playground/gad_rgd1.py --eigen-method svdforce
python playground/gad_rgd1.py --eigen-method inertia
python playground/gad_rgd1.py --eigen-method geo
python playground/gad_rgd1.py --eigen-method ase
python playground/gad_rgd1.py --eigen-method eckartsvd
python playground/gad_rgd1.py --eigen-method eckartqr
"""

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
import argparse
import json

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.equiformer_calculator import EquiformerCalculator

# from gadff.align_unordered_mols import rmsd
from gadff.align_ordered_mols import align_ordered_and_get_rmsd
from gadff.plot_molecules import (
    plot_molecule_mpl,
    plot_traj_mpl,
    clean_filename,
    save_to_xyz,
    save_trajectory_to_xyz,
)
from gadff.align_ordered_mols import find_rigid_alignment

from ase import Atoms
from ase.io import read
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from sella import Sella, Constraints
from gadff.equiformer_ase_calculator import EquiformerASECalculator

this_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(this_dir, "plots_gad")
log_dir = os.path.join(this_dir, "logs_gad")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


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


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


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
    n_patience_steps=50,
    patience_threshold=2.0,
    center_around_com=False,
    align_rotation=False,
):
    assert force_field in ["gad", "forces"], f"Unknown force field: {force_field}"

    device = calc.model.device

    title += f" dt{dt}"

    print("")
    print("-" * 6)
    print(f"Following {force_field} vector field: {title}")

    # Compute RMSD between initial guess and transition state
    rmsd_initial = align_ordered_and_get_rmsd(initial_pos.numpy(), true_pos.numpy())
    print(f"RMSD start: {rmsd_initial:.6f} Å")

    # Create initial batch from interpolated guess
    current_pos = initial_pos.clone().requires_grad_(True).to(device)

    # Storage for trajectory
    trajectory_pos = []
    trajectory_energy = []
    trajectory_forces_norm = []
    trajectory_forces = []

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
        else:
            raise ValueError(f"Unknown force field: {force_field}")
        forces = forces.detach()
        energy = energy.detach()

        # Store trajectory
        trajectory_pos.append(current_pos.detach().cpu().clone())
        trajectory_forces.append(forces.cpu())
        trajectory_energy.append(energy.item())

        # Check for convergence (small vector magnitude)
        forces_norm = torch.norm(forces).item()
        trajectory_forces_norm.append(forces_norm)

        if forces_norm < convergence_threshold:
            print(f"Converged at step {step}, forces norm: {forces_norm:.6f}")
            break

        # terminate if force norm was small for 50 steps
        if np.all(
            np.array(trajectory_forces_norm[-n_patience_steps:]) < patience_threshold
        ):
            print(f"Force norm was small for {n_patience_steps} steps, terminating")
            break

        # Euler integration step
        current_pos = current_pos.detach() + dt * forces  # [N, 3]

        # center around center of mass
        if center_around_com:
            current_pos = current_pos - current_pos.mean(dim=0, keepdim=True)

        if align_rotation:
            # align rotation of current_pos to initial_pos
            _rot, _trans = find_rigid_alignment(current_pos, initial_pos)
            current_pos = (current_pos @ _rot.T) + _trans

        if step % print_every == 0:
            print(
                f"Step {step}: VF norm = {forces_norm:.6f}, Energy = {energy.item():.6f}"
            )

    # Final position
    final_pos = trajectory_pos[-1]

    steps = len(trajectory_pos)
    title += f" s{steps}"
    title = f"{force_field} - {title}"

    _this_plot_dir = os.path.join(plot_dir, clean_filename(title, None, None))
    os.makedirs(_this_plot_dir, exist_ok=True)

    # After convergence, compute final RMSD and eigenvalues
    # Plot trajectory
    plot_traj_mpl(
        coords_traj=trajectory_pos,
        title=title,
        plot_dir=_this_plot_dir,
        atomic_numbers=z,
        save=True,
        # forces_traj=trajectory_gad,
    )

    # plot final position
    plot_molecule_mpl(
        final_pos,
        atomic_numbers=z,
        title=title,
        plot_dir=_this_plot_dir,
        save=True,
    )

    # save xyz to visualize with Mol* / Protein Viewer
    save_to_xyz(trajectory_pos[-1], z, plotfolder=_this_plot_dir, filename=title)
    save_trajectory_to_xyz(trajectory_pos, z, plotfolder=_this_plot_dir, filename=title)

    # Compute RMSD between converged structure and true transition state
    rmsd_final = align_ordered_and_get_rmsd(final_pos.numpy(), true_pos.numpy())
    print(f"RMSD end: {rmsd_final:.6f} Å")
    print(f"Improvement: {rmsd_initial - rmsd_final:.6f} Å")

    # Compute Hessian eigenvalues at final structure
    batch_final = Batch.from_data_list(
        [TGData(z=z, pos=final_pos.requires_grad_(True), natoms=natoms)]
    )

    # Computing Hessian eigenvalues at converged structure
    energy_final, forces_final, hessian, eigenvalues, eigenvectors, eigenpred = (
        calc.predict_with_hessian(batch_final)
    )

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
    fig_name = clean_filename(title, "convergence")
    plt.savefig(os.path.join(_this_plot_dir, fig_name), dpi=150, bbox_inches="tight")
    print(f"Saved convergence plot to {os.path.join(_this_plot_dir, fig_name)}")
    return trajectory_pos, trajectory_energy, trajectory_forces_norm, trajectory_forces


def run_sella(
    pos_initial_guess,
    z,
    natoms,
    true_pos,
    title,
    calc=None,
    hessian_function=None,
    sella_kwargs={},
):
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

    print("")
    print("-" * 6)
    print(f"Starting Sella {calcname} optimization to find transition state")

    true_pos = to_numpy(true_pos)
    rmsd_initial = align_ordered_and_get_rmsd(positions_np, true_pos)
    print(f"RMSD start: {rmsd_initial:.6f} Å")

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
        hessian_function=hessian_function,
        **sella_kwargs,
    )

    # Run with much tighter convergence
    dyn.run(1e-5, 2000)  # Much stricter convergence criterion
    print("Sella optimization completed!")

    # Get the final structure, convert to np, compute RMSD to true transition state
    final_positions_sella = mol_ase.get_positions()

    # Compute RMSD between Sella-optimized structure and true transition state
    rmsd_sella = align_ordered_and_get_rmsd(final_positions_sella, true_pos)
    print(f"RMSD end: {rmsd_sella:.6f} Å")
    print(f"Improvement: {rmsd_initial - rmsd_sella:.6f} Å")

    _this_plot_dir = os.path.join(plot_dir, clean_filename(title, None, None))
    os.makedirs(_this_plot_dir, exist_ok=True)

    traj = dyn.trajectory
    plot_traj_mpl(
        coords_traj=traj,
        title=f"Sella {calcname} {title}",
        plot_dir=_this_plot_dir,
        atomic_numbers=z,
        save=True,
    )
    # plot_molecule_mpl(
    #     mol_ase.get_positions(),
    #     atomic_numbers=z,
    #     title=f"Sella {calcname} {title}",
    #     plot_dir=_this_plot_dir,
    #     save=True,
    # )
    save_trajectory_to_xyz(traj, z, plotfolder=_this_plot_dir, filename=title)
    return mol_ase


def example(
    eigen_method="qr",
):
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

    # Initialize equiformer calculator
    print("\n" + "-" * 6)
    print(f"Initializing EquiformerCalculator with eigen_method={eigen_method}")
    calc = EquiformerCalculator(device=device, eigen_dof_method=eigen_method)

    # # Create batch (equiformer expects batch format)
    # tg_data = convert_rgd1_to_tg_format(sample, state="transition")
    # batch = Batch.from_data_list([tg_data])
    # print(f"Batch shape - pos: {batch.pos.shape}, z: {batch.z.shape}")
    # # Run prediction
    # print("\n" + "-" * 6)
    # energy, forces, eigenpred = calc.predict(batch)
    # print(f"Energy: {energy.item():.6f}")
    # print(f"Forces shape: {forces.shape}")
    # print(f"Forces norm: {torch.norm(forces).item():.6f}")
    # print(f"Eigenprediction keys: {list(eigenpred.keys())}")

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
    print("\n" + "=" * 60)
    print("Following GAD vector field to find transition state")

    results = {}

    # Test 1: is the transition state a fixed point of our GAD vector field?
    # Follow the GAD vector field from transition state
    traj, _, _, _ = integrate_dynamics(
        sample.pos_transition,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=100,
        dt=0.01,
        title=f"TS from TS {eigen_method}",
        n_patience_steps=1000,
        patience_threshold=1.0,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_ts"] = _rmsd_ts

    # Follow the GAD vector field from perturbed transition state
    _pos = (
        torch.randn_like(sample.pos_transition) + sample.pos_transition
    )  # RMSD ~ 1.2 Å
    traj, _, _, _ = integrate_dynamics(
        _pos,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=1000,
        dt=0.01,
        title=f"TS from perturbed TS {eigen_method}",
        n_patience_steps=1000,
        patience_threshold=1.0,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_perturbed_ts"] = _rmsd_ts

    # Test run - start from reactant
    traj, _, _, _ = integrate_dynamics(
        pos_reactant,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=100,
        dt=0.1,
        title=f"TS from R {eigen_method}",
        n_patience_steps=100,
        # patience_threshold=1.0,
        # center_around_com=True,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.1_s100"] = _rmsd_ts

    # Start from reactant
    traj, _, _, _ = integrate_dynamics(
        pos_reactant,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=1_000,
        dt=0.01,
        title=f"TS from R {eigen_method}",
        n_patience_steps=1000,
        # patience_threshold=1.0,
        # center_around_com=True,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.01_s1000"] = _rmsd_ts

    # large steps
    traj, _, _, _ = integrate_dynamics(
        pos_reactant,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=1_000,
        dt=0.1,
        title=f"TS from R {eigen_method}",
        n_patience_steps=1000,
        # patience_threshold=1.0,
        # center_around_com=True,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.1_s1000"] = _rmsd_ts

    # very long
    traj, _, _, _ = integrate_dynamics(
        pos_reactant,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=10_000,
        dt=0.01,
        title=f"TS from R (10k steps) {eigen_method}",
        n_patience_steps=10000,
        # patience_threshold=1.0,
        # center_around_com=True,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.1_s10000"] = _rmsd_ts

    # Follow the GAD vector field from R-P interpolation
    traj, _, _, _ = integrate_dynamics(
        pos_initial_guess_rp,
        sample.z,
        sample.natoms,
        calc,
        sample.pos_transition,
        force_field="gad",
        max_steps=1_000,
        dt=0.01,
        title=f"R-P interpolation {eigen_method}",
        n_patience_steps=500,
        # patience_threshold=1.0,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_p_dt0.01_s1000"] = _rmsd_ts

    # Save results to JSON
    with open(os.path.join(log_dir, f"results_{eigen_method}.json"), "w") as f:
        json.dump(results, f)

    with open(os.path.join(log_dir, f"results_{eigen_method}.txt"), "w") as f:
        f.write(f"# eigen_method: {eigen_method}\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    return results

    ###################################################################################
    print("=" * 60)
    print("Following Sella to find transition state")

    # See if Sella can find the transition state
    asecalc = EquiformerASECalculator(device=device)

    # Linear interpolation between reactant and product
    mol_ase = run_sella(
        pos_initial_guess_rp,
        z=sample.z,
        natoms=sample.natoms,
        true_pos=sample.pos_transition,
        title="TS from R-P",
        calc=asecalc,
    )

    # Linear interpolation between reactant and TS
    mol_ase = run_sella(
        pos_initial_guess_rts,
        z=sample.z,
        natoms=sample.natoms,
        true_pos=sample.pos_transition,
        title="TS from R-TS",
        calc=asecalc,
    )

    # Start from reactant
    mol_ase = run_sella(
        pos_reactant,
        z=sample.z,
        natoms=sample.natoms,
        true_pos=sample.pos_transition,
        title="TS from R",
        calc=asecalc,
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
    # Test Sella with Equiformer
    print("=" * 60)
    print("Testing Sella with Equiformer Hessian")

    # asecalchessian = EquiformerASECalculator(checkpoint_path="model.pt")
    # Create wrapper function
    def ml_hessian_function(atoms):
        return asecalc.get_hessian_autodiff(atoms)

    mol_ase = run_sella(
        pos_initial_guess_rp,
        z=sample.z,
        natoms=sample.natoms,
        true_pos=sample.pos_transition,
        title="Autodiff Hessian Cartesian",
        calc=asecalc,
        hessian_function=ml_hessian_function,
    )
    mol_ase = run_sella(
        pos_initial_guess_rp,
        z=sample.z,
        natoms=sample.natoms,
        true_pos=sample.pos_transition,
        title="Autodiff Hessian Internal",
        calc=asecalc,
        hessian_function=ml_hessian_function,
        sella_kwargs={"internal": True},
    )

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
    parser = argparse.ArgumentParser(description="Run GAD-RGD1 example.")
    parser.add_argument(
        "--eigen-method",
        type=str,
        default="qr",
        help="Eigenvalue method for GAD (qr, svd, svdforce, inertia, geo, ase, eckartsvd, eckartqr)",
    )

    args, unknown = parser.parse_known_args()

    # eigen_kwargs = {}
    # # Parse any additional arguments in format --key=value and add to eigen_kwargs
    # for arg in unknown:
    #     if arg.startswith('--') and '=' in arg:
    #         key, value = arg[2:].split('=', 1)
    #         # Try to convert to appropriate type
    #         try:
    #             # Try float first
    #             value = float(value)
    #         except ValueError:
    #             try:
    #                 # Try int
    #                 value = int(value)
    #             except ValueError:
    #                 # Keep as string
    #                 pass
    #         eigen_kwargs[key] = value
    #     elif arg.startswith('--'):
    #         # Boolean flag
    #         key = arg[2:]
    #         eigen_kwargs[key] = True

    results = example(eigen_method=args.eigen_method)
    print("Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")
