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
import inspect
from typing import Callable

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
    plot_dir=None,
):
    assert force_field in ["gad", "forces"], f"Unknown force field: {force_field}"

    if plot_dir is None:
        # Auto-detect plot directory based on calling file
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        plot_dir = os.path.join(caller_dir, "plots")

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


def before_ase(start_pos, z, true_pos=None, calc=None):
    # Create ASE Atoms object from the initial guess position
    start_pos = to_numpy(start_pos)
    atomic_numbers_np = to_numpy(z)
    mol_ase = Atoms(numbers=atomic_numbers_np, positions=start_pos)
    if calc is None:
        mol_ase.calc = EMT()
    else:
        mol_ase.calc = calc
    calcname = calc.__class__.__name__

    # Measure RMSD between start_pos and true_pos
    if true_pos is not None:
        true_pos = to_numpy(true_pos)
        rmsd_initial = align_ordered_and_get_rmsd(start_pos, true_pos)
        print(f"RMSD start: {rmsd_initial:.6f} Å")
    
    summary = {
        "rmsd_initial": rmsd_initial,
        "calcname": calcname,
    }
    return mol_ase, summary

def after_ase(summary, z, title, true_pos=None, plot_dir=None):
    if plot_dir is None:
        # Auto-detect plot directory based on calling file
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        plot_dir = os.path.join(caller_dir, "plots")
        print(f"Autodetected plot directory: {plot_dir}")
    trajectory = summary.get("trajectory", None)
    rmsd_initial = summary.get("rmsd_initial", float("inf"))
    calcname = summary.get("calcname", "")
    endsummary = {}
    if trajectory is not None and true_pos is not None:
        pred_pos = trajectory[-1]

        # Compute RMSD between Sella-optimized structure and true transition state
        rmsd_final = align_ordered_and_get_rmsd(pred_pos, true_pos)
        print(f"RMSD end: {rmsd_final:.6f} Å")
        print(f"Improvement: {rmsd_initial - rmsd_final:.6f} Å")
        endsummary["rmsd_final"] = rmsd_final
        endsummary["rmsd_improvement"] = rmsd_initial - rmsd_final

    _this_plot_dir = os.path.join(plot_dir, clean_filename(title, None, None))
    os.makedirs(_this_plot_dir, exist_ok=True)

    # plot trajectory
    if len(trajectory) > 0:
        plot_traj_mpl(
            coords_traj=trajectory,
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
        save_trajectory_to_xyz(trajectory, z, plotfolder=_this_plot_dir, filename=title)
    return endsummary
    
def run_with_ase_wrapper(run_fn: Callable, start_pos, z, title, true_pos=None, calc=None, plot_dir=None):
    # build atoms object
    mol_ase, initsummary = before_ase(start_pos, z, true_pos, calc)
    # run function
    summary = run_fn(mol_ase)
    summary.update(initsummary)
    # eval and plot
    endsummary = after_ase(summary, z, title, true_pos, plot_dir)
    summary.update(endsummary)
    return summary


def run_sella(
    start_pos,
    z,
    natoms,
    true_pos,
    title,
    calc=None,
    hessian_function=None,
    sella_kwargs={},
    plot_dir=None,
):
    if plot_dir is None:
        # Auto-detect plot directory based on calling file
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        plot_dir = os.path.join(caller_dir, "plots")
        print(f"Sella autodetected plot directory: {plot_dir}")

    mol_ase, initsummary = before_ase(start_pos, z, true_pos, calc)

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
    dyn.run(fmax=1e-5, steps=2000)  # Much stricter convergence criterion
    print("Sella optimization completed!")

    # Get the final structure, convert to np, compute RMSD to true transition state
    final_positions_sella = mol_ase.get_positions()
    traj = dyn.trajectory
    
    summary = {
        "trajectory": traj,
    }
    summary.update(initsummary)

    endsummary = after_ase(summary, z, title, true_pos, plot_dir=None)
    summary.update(endsummary)
    return summary