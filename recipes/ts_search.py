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
import sys
from typing import Callable, Literal
import copy
import logging
import math
import time
from datetime import datetime
import traceback

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.equiformer_calculator import EquiformerTorchCalculator

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
from ase.vibrations.data import VibrationsData
from ase.vibrations import Vibrations
from ase.optimize import BFGS
from ase.mep import NEB
from sella import Sella, Constraints, IRC
from recipes.trajectorysaver import MyTrajectory
from gadff.equiformer_ase_calculator import EquiformerASECalculator

# https://github.com/virtualzx-nad/geodesic-interpolate
from geodesic_interpolate.geodesic import Geodesic
from geodesic_interpolate.interpolation import redistribute


def clean_dict(data):
    """
    Recursively clean a dictionary to contain only JSON-serializable types.

    - Keeps: int, float, str, bool, None
    - Converts single-element numpy arrays and torch tensors to scalars
    - Discards multi-element arrays/tensors and other non-serializable types
    - Recursively processes nested dictionaries and lists

    Args:
        data: Input data structure to clean

    Returns:
        Cleaned data structure with only serializable types
    """
    if data is None:
        return None
    elif isinstance(data, (int, float, str, bool)):
        return data
    elif isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            cleaned_value = clean_dict(value)
            if cleaned_value is not None or value is None:
                cleaned[str(key)] = cleaned_value
        return cleaned
    elif isinstance(data, (list, tuple)):
        cleaned = []
        for item in data:
            cleaned_item = clean_dict(item)
            if cleaned_item is not None or item is None:
                cleaned.append(cleaned_item)
        return cleaned
    elif torch.is_tensor(data):
        # Convert single-element tensors to scalars, discard multi-element
        if data.numel() == 1:
            return data.item()
        else:
            return None
    elif isinstance(data, np.ndarray):
        # Convert single-element arrays to scalars, discard multi-element
        if data.size == 1:
            return data.item()
        else:
            return None
    elif hasattr(data, "item") and hasattr(data, "size"):
        # Handle other array-like objects with single elements
        try:
            if data.size == 1:
                return data.item()
        except:
            pass
        return None
    elif isinstance(data, type):
        # Return class name for classes
        return data.__class__.__name__
    elif hasattr(data, "__name__"):
        # Return function name for functions
        return data.__name__
    # elif callable(data):
    #     return
    else:
        # Discard other non-serializable types
        return None


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


def copy_atoms(atoms: Atoms) -> Atoms:
    """
    Simple function to copy an atoms object to prevent mutability.
    """
    # try:
    #     atoms = copy.deepcopy(atoms)
    # except Exception:
    #     # Needed because of ASE issue #1084
    calc = atoms.calc
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x).to(device)


def get_hessian_function(hessian_method, asecalc):
    if hessian_method == "autodiff":

        def _hessian_function(atoms):
            return asecalc.get_hessian_autodiff(atoms).reshape(
                (len(atoms) * 3, len(atoms) * 3)
            )

    elif hessian_method == "predict":

        def _hessian_function(atoms):
            return asecalc.get_hessian_prediction(atoms).reshape(
                (len(atoms) * 3, len(atoms) * 3)
            )

    elif hessian_method is None:
        _hessian_function = None
    else:
        raise ValueError(f"Invalid method: {hessian_method}")
    return _hessian_function


########################################################################################
# Optimization


def integrate_dynamics(
    initial_pos: torch.Tensor,
    z: torch.Tensor,
    natoms: torch.Tensor,
    torchcalc,
    true_pos: torch.Tensor,
    force_field="gad",
    title=None,
    max_steps=3_000,
    convergence_threshold=1e-4,
    dt=0.01,
    print_every=100,
    n_patience_steps=1_000,
    patience_threshold=2.0,
    center_around_com=False,
    align_rotation=False,
    plot_dir=None,
    do_freq=True,
    asecalc=None,
):
    assert force_field in ["gad", "forces"], f"Unknown force field: {force_field}"

    if plot_dir is None:
        # Auto-detect plot directory based on main script
        main_script = sys.argv[0]
        main_dir = os.path.dirname(os.path.abspath(main_script))
        plot_dir = os.path.join(main_dir, "plots")

    title += f" dt{dt}"

    print("")
    print("-" * 6)
    print(f"Following {force_field} vector field: {title}")

    # Compute RMSD between initial guess and transition state
    rmsd_initial = align_ordered_and_get_rmsd(initial_pos, true_pos)
    print(f"RMSD start: {rmsd_initial:.6f} Å")

    device = torchcalc.model.device
    initial_pos = to_torch(initial_pos, device)
    true_pos = to_torch(true_pos, device)
    # z = to_torch(z, device)
    # natoms = to_torch(natoms, device)

    # Create initial batch from interpolated guess
    current_pos = initial_pos.clone().requires_grad_(True).to(device)

    # Storage for trajectory
    trajectory_pos = []
    trajectory_energy = []
    trajectory_forces_norm = []
    trajectory_forces = []

    t1 = time.time()
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
            gad_vector, out = torchcalc.gad_autograd_hessian(batch_current)
            energy = out["energy"]
            forces = gad_vector.reshape(-1, 3)
        elif force_field == "forces":
            # Compute forces
            energy, forces, eigenpred = torchcalc.predict(batch_current)
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
        if step > n_patience_steps:
            if np.all(
                np.array(trajectory_forces_norm[-n_patience_steps:])
                < patience_threshold
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
    t2 = time.time()
    steps = len(trajectory_pos)
    print(f"Time taken: {t2 - t1:.2f} seconds (steps={steps})")
    # Final position
    final_pos = trajectory_pos[-1]

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
    rmsd_final = align_ordered_and_get_rmsd(final_pos, true_pos)
    print(f"RMSD end: {rmsd_final:.6f} Å")
    print(f"Improvement: {rmsd_initial - rmsd_final:.6f} Å")

    # Compute Hessian eigenvalues at final structure
    batch_final = Batch.from_data_list(
        [TGData(z=z, pos=final_pos.clone().requires_grad_(True), natoms=natoms)]
    )

    # Computing Hessian eigenvalues at converged structure
    energy_final, forces_final, hessian, eigenvalues, eigenvectors, eigenpred = (
        torchcalc.predict_with_hessian(batch_final)
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

    summary = {
        "trajectory": trajectory_pos,
        "energy": trajectory_energy,
        "forces_norm": trajectory_forces_norm,
        "forces": trajectory_forces,
        "rmsd_initial": rmsd_initial,
        "rmsd_final": rmsd_final,
        "rmsd_improvement": rmsd_initial - rmsd_final,
        "time_taken": t2 - t1,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if do_freq:
        mol_ase = Atoms(numbers=z, positions=final_pos)
        mol_ase.calc = asecalc
        freq_summary = run_freq(mol_ase, asecalc)
        summary.update(freq_summary)
    return summary


def before_ase_opt(start_pos, z, true_pos=None, calc=None):
    # Create ASE Atoms object from the initial guess position
    start_pos = to_numpy(start_pos)
    atomic_numbers_np = to_numpy(z)
    mol_ase = Atoms(numbers=atomic_numbers_np, positions=start_pos)
    if calc is None:
        mol_ase.calc = EMT()
    else:
        mol_ase.calc = calc
    calcname = calc.__class__.__name__

    summary = {
        "calcname": calcname,
    }

    # Measure RMSD between start_pos and true_pos
    if true_pos is not None:
        true_pos = to_numpy(true_pos)
        rmsd_initial = align_ordered_and_get_rmsd(start_pos, true_pos)
        print(f"RMSD start: {rmsd_initial:.6f} Å")
        summary["rmsd_initial"] = rmsd_initial
    return mol_ase, summary


def after_ase_opt(summary, z, title, true_pos=None, plot_dir=None):
    """Computes RMSD between predicted and true transition state, and plots trajectory"""
    if plot_dir is None:
        # Auto-detect plot directory based on main script
        main_script = sys.argv[0]
        main_dir = os.path.dirname(os.path.abspath(main_script))
        plot_dir = os.path.join(main_dir, "plots")
        print(f"Autodetected plot directory: {plot_dir}")
    trajectory = summary.get("trajectory", None)
    rmsd_initial = summary.get("rmsd_initial", float("inf"))
    calcname = summary.get("calcname", "")
    endsummary = {}
    if (trajectory is not None) and (true_pos is not None):
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
            title=title,
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

    # save summary dict
    summary.update(endsummary)
    cleansummary = clean_dict(summary)
    with open(os.path.join(_this_plot_dir, "summary.json"), "w") as f:
        json.dump(cleansummary, f)
    print(f"Saved summary to {os.path.join(_this_plot_dir, 'summary.json')}")
    with open(os.path.join(_this_plot_dir, "summary.txt"), "w") as f:
        f.write(json.dumps(cleansummary, indent=4))
    print(f"Saved summary to {os.path.join(_this_plot_dir, 'summary.txt')}")
    return endsummary


def run_with_ase_wrapper(
    run_fn: Callable, start_pos, z, title, true_pos=None, calc=None, plot_dir=None
):
    # build atoms object
    mol_ase, initsummary = before_ase_opt(start_pos, z, true_pos, calc)
    # run function
    summary = run_fn(mol_ase)
    summary.update(initsummary)
    # eval and plot
    summary = after_ase_opt(summary, z, title, true_pos, plot_dir)
    return summary


sella_default_kwargs = dict(
    minimum=dict(
        delta0=1e-1,
        sigma_inc=1.15,
        sigma_dec=0.90,
        rho_inc=1.035,
        rho_dec=100,
        method="rfo",
        eig=False,
    ),
    saddle=dict(
        delta0=0.1,
        sigma_inc=1.15,
        sigma_dec=0.65,
        rho_inc=1.035,
        rho_dec=5.0,
        method="prfo",
        eig=True,
    ),
)


def run_sella(
    start_pos,
    z,
    natoms,
    true_pos,
    title,
    calc=None,
    hessian_function=None,
    hessian_method=None,
    sella_kwargs={},
    run_kwargs={},
    diag_every_n=None,  # brute force. paper set this to 1. try 0
    nsteps_per_diag=3,  # adaptive
    internal=False,  # paper set this to True
    plot_dir=None,
    do_freq=True,
):
    if hessian_function is None:
        hessian_method = None
    else:
        if hessian_method is None:
            hessian_method = hessian_function.__name__
    title += f" | Hessian={hessian_method}"
    if internal:
        title += " | Internal"
    else:

        title += " | Cartesian"
    if diag_every_n is not None:
        title += f" | diag_every_n={diag_every_n}"
    # if nsteps_per_diag != 3:
    #     title += f" | nsteps_per_diag={nsteps_per_diag}"
    print("\nRunning Sella TS search:", title)
    if plot_dir is None:
        # Auto-detect plot directory based on main script
        main_script = sys.argv[0]
        main_dir = os.path.dirname(os.path.abspath(main_script))
        plot_dir = os.path.join(main_dir, "plots_sella")
        print(f"Sella autodetected plot directory: {plot_dir}")

    mol_ase, initsummary = before_ase_opt(start_pos, z, true_pos, calc)

    # Set up a Sella Dynamics object with improved parameters for TS search
    _sella_kwargs = dict(
        atoms=mol_ase,
        # constraints=cons,
        # trajectory=os.path.join(logfolder, "cu_sella_ts.traj"),
        trajectory=MyTrajectory(atoms=mol_ase),
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
        diag_every_n=diag_every_n,
        nsteps_per_diag=nsteps_per_diag,
        internal=internal,
    )
    _sella_kwargs.update(sella_kwargs)
    dyn = Sella(
        **_sella_kwargs,
    )

    _run_kwargs = dict(
        fmax=1e-3,
        steps=4000,
    )
    _run_kwargs.update(run_kwargs)

    summary = {
        "sella_kwargs": _sella_kwargs,
        "run_kwargs": _run_kwargs,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Run the optimization
    t1 = time.time()
    try:
        dyn.run(**_run_kwargs)
    except Exception as e:
        error_traceback = traceback.print_exc()
        print(f"Error in Sella optimization: {e}")
        print(f"Sella kwargs: {_sella_kwargs}")
        print(f"Run kwargs: {_run_kwargs}")
        print("Full stack trace:")
        print(error_traceback)
        summary["error"] = str(e)
        summary["error_traceback"] = error_traceback
        return summary
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")
    print("Sella optimization completed!")

    # Get the final structure, convert to np, compute RMSD to true transition state
    # final_positions_sella = mol_ase.get_positions()
    # traj = dyn.trajectory.trajectory # ASE
    traj = dyn.pes.traj.trajectory  # Sella PES trajectory

    summary.update(
        {
            "trajectory": traj,
            "nsteps": dyn.nsteps,
            "time_taken": t2 - t1,
        }
    )
    summary.update(initsummary)

    if do_freq:
        freq_summary = run_freq(mol_ase, calc)
        summary.update(freq_summary)

    endsummary = after_ase_opt(summary, z, title, true_pos, plot_dir=plot_dir)
    summary.update(endsummary)
    return summary


def run_freq(atoms, calc):
    summary = {}
    for freq_method in ["autodiff", "predict", "finite_diff"]:
        final_atoms = copy_atoms(atoms)
        t1 = time.time()
        if freq_method == "autodiff":
            hessian = calc.get_hessian_autodiff(final_atoms)
            vib_data = VibrationsData.from_2d(final_atoms, hessian)
            # # Modes are given in Cartesian coordinates as a (3N, N, 3) array
            # # where indices correspond to the (mode_index, atom, direction)
            # energies, modes = vib.get_energies_and_modes()
            # frequencies = energies / ase.units.invcm # cm-1
        elif freq_method == "predict":
            hessian = calc.get_hessian_prediction(final_atoms)
            vib_data = VibrationsData.from_2d(final_atoms, hessian)
        elif freq_method == "finite_diff":
            vib = Vibrations(final_atoms)
            vib.run()
            vib_data = vib.get_vibrations()
        else:
            raise ValueError(f"Unknown frequency method: {freq_method}")
        t2 = time.time()
        print(f"Time taken for {freq_method}: {t2 - t1:.2f} seconds")
        # Get vibrational data
        frequencies = vib_data.get_frequencies()  # in cm^-1
        energies = vib_data.get_energies()  # in eV
        modes = vib_data.get_modes()

        # Check if this is a transition state (exactly one negative frequency)
        negative_freq_count = np.sum(frequencies < 0)
        is_transition_state = negative_freq_count == 1
        print(f"Is transition state: {is_transition_state} for {freq_method}")

        summary[f"freq_{freq_method}"] = frequencies
        summary[f"energy_{freq_method}"] = energies
        summary[f"mode_{freq_method}"] = modes
        summary[f"negative_freq_count_{freq_method}"] = negative_freq_count
        summary[f"is_transition_state_{freq_method}"] = is_transition_state
        summary[f"time_taken_{freq_method}"] = t2 - t1
        summary[f"date_{freq_method}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # vib.summary()
        # summary_lines = VibrationsData._tabulate_from_energies(energies)
        # log_text = '\n'.join(summary_lines) + '\n'
    return summary


def run_irc(
    start_pos,
    z,
    true_pos,
    title,
    calc=None,
    opt_kwargs={},
    run_kwargs={},
):

    atoms, initsummary = before_ase_opt(start_pos, z, true_pos, calc)

    _opt_kwargs = dict(
        dx=0.1,
        eta=1e-4,
        gamma=0.4,
        keep_going=True,
        trajectory=MyTrajectory(atoms=atoms),
    )
    _opt_kwargs.update(opt_kwargs)
    dyn = IRC(atoms, **_opt_kwargs)
    _run_kwargs = dict(
        # fmax=1e-5, steps=2000,
    )
    _run_kwargs.update(run_kwargs)
    t1 = time.time()
    dyn.run(direction="forward")
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")
    traj = dyn.trajectory.trajectory
    summary = {
        "trajectory": traj,
        "nsteps": dyn.nsteps,
        "time_taken": t2 - t1,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    endsummary = after_ase_opt(summary, z, title, true_pos, plot_dir=None)
    summary.update(initsummary)
    summary.update(endsummary)
    return summary


def _run_relaxation(
    atoms: Atoms,
    method="sella",
    opt_kwargs={},
    run_kwargs={},
    do_freq=False,
):

    _run_kwargs = dict(steps=10_000)
    _run_kwargs.update(run_kwargs)
    if method == "bfgs":
        dyn = BFGS(atoms, trajectory=MyTrajectory(atoms=atoms))
        t1 = time.time()
        dyn.run(**_run_kwargs)
        t2 = time.time()
        print(f"Time taken: {t2 - t1:.2f} seconds")
        traj = dyn.trajectory.trajectory
    elif method == "sella":
        _opt_kwargs = dict(
            order=0,
            internal=False,
        )
        _opt_kwargs.update(opt_kwargs)
        dyn = Sella(
            atoms,
            trajectory=MyTrajectory(atoms=atoms),
            **_opt_kwargs,
        )
        t1 = time.time()
        dyn.run(**_run_kwargs)
        t2 = time.time()
        print(f"Time taken: {t2 - t1:.2f} seconds")
        traj = dyn.pes.traj.trajectory
    else:
        raise ValueError(f"Unknown relaxation method: {method}")
    summary = {
        "atoms": atoms,
        "trajectory": traj,
        "nsteps": dyn.nsteps,
        "time_taken": t2 - t1,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary


def run_relaxation(
    start_pos=None,
    z=None,
    true_pos=None,
    calc=None,
    title=None,
    method="sella",
    opt_kwargs={},
    run_kwargs={},
    do_freq=False,
    plot_dir=None,
):
    atoms, initsummary = before_ase_opt(start_pos, z, true_pos, calc)
    summary = _run_relaxation(atoms, method, opt_kwargs, run_kwargs)
    initsummary.update(summary)
    if do_freq:
        freq_summary = run_freq(atoms, calc)
        initsummary.update(freq_summary)
    endsummary = after_ase_opt(initsummary, z, title, true_pos, plot_dir=None)
    initsummary.update(endsummary)
    return summary


def run_neb(
    reactant_atoms,
    product_atoms,
    z,
    true_pos,
    calc,
    relax_opt_kwargs={},
    relax_run_kwargs={},
    interpolation_method="geodesic",
    interpolation_kwargs={},
    neb_kwargs={},
    do_freq=False,
    plot_dir=None,
    title=None,
    do_freq_relax=False,
):
    _interpolation_kwargs = dict(n_images=20)
    _interpolation_kwargs.update(interpolation_kwargs)
    _neb_kwargs = dict(method="aseneb", precon=None)
    _neb_kwargs.update(neb_kwargs)

    reactant_atoms, summary_r = before_ase_opt(reactant_atoms, z, true_pos, calc)
    product_atoms, summary_p = before_ase_opt(product_atoms, z, true_pos, calc)

    # Run relax job
    relax_summary_r = _run_relaxation(
        reactant_atoms,
        opt_kwargs=relax_opt_kwargs,
        run_kwargs=relax_run_kwargs,
        do_freq=do_freq_relax,
    )
    relax_summary_p = _run_relaxation(
        product_atoms,
        opt_kwargs=relax_opt_kwargs,
        run_kwargs=relax_run_kwargs,
        do_freq=do_freq_relax,
    )

    if interpolation_method == "geodesic":
        images = geodesic_interpolate_wrapper(
            relax_summary_r["atoms"], relax_summary_p["atoms"], **_interpolation_kwargs
        )
    else:
        images = [reactant_atoms]
        images += [
            reactant_atoms.copy() for _ in range(_interpolation_kwargs["n_images"] - 2)
        ]
        images += [product_atoms]
        neb = NEB(images)

        # Interpolate linearly the positions of the middle images:
        neb.interpolate(method=interpolation_method)
        images = neb.images

    summary = {
        "relax_reactant": relax_summary_r,
        "relax_product": relax_summary_p,
        "initial_images": copy_atoms(images),
    }

    max_steps = _interpolation_kwargs.pop("max_steps", None)
    dyn = NEB(images)
    t1 = time.time()
    dyn.run(max_steps=max_steps, neb_kwargs=_neb_kwargs)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")

    # summary["neb_trajectory"] = dyn.trajectory.trajectory
    summary["neb_summary"] = dyn.summary
    summary["nsteps"] = dyn.nsteps
    summary["time_taken"] = t2 - t1
    summary["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if do_freq:
        freq_summary = run_freq(images, calc)
        summary.update(freq_summary)

    endsummary = after_ase_opt(summary, z, title, true_pos, plot_dir=None)
    summary.update(endsummary)
    return summary


def geodesic_interpolate_wrapper(
    reactant: Atoms,
    product: Atoms,
    n_images: int = 10,
    perform_sweep: bool | Literal["auto"] = "auto",
    redistribute_tol: float = 1e-2,
    smoother_tol: float = 2e-3,
    max_iterations: int = 15,
    max_micro_iterations: int = 20,
    morse_scaling: float = 1.7,
    geometry_friction: float = 1e-2,
    distance_cutoff: float = 3.0,
    sweep_cutoff_size: int = 35,
) -> list[Atoms]:
    """
    Interpolates between two geometries and optimizes the path with the geodesic method.

    Parameters
    ----------
    reactant
        The ASE Atoms object representing the initial geometry.
    product
        The ASE Atoms object representing the final geometry.
    n_images
        Number of images for interpolation. Default is 10.
    perform_sweep
        Whether to sweep across the path optimizing one image at a time.
        Default is to perform sweeping updates if there are more than 35 atoms.
    redistribute_tol
        the value passed to the tol keyword argument of
         geodesic_interpolate.interpolation.redistribute. Default is 1e-2.
    smoother_tol
        the value passed to the tol keyword argument of geodesic_smoother.smooth
        or geodesic_smoother.sweep. Default is 2e-3.
    max_iterations
        Maximum number of minimization iterations. Default is 15.
    max_micro_iterations
        Maximum number of micro iterations for the sweeping algorithm. Default is 20.
    morse_scaling
        Exponential parameter for the Morse potential. Default is 1.7.
    geometry_friction
        Size of friction term used to prevent very large changes in geometry. Default is 1e-2.
    distance_cutoff
        Cut-off value for the distance between a pair of atoms to be included in the coordinate system. Default is 3.0.
    sweep_cutoff_size
        Cut off system size that above which sweep function will be called instead of smooth
        in Geodesic.

    Returns
    -------
    list[Atoms]
        A list of ASE Atoms objects representing the smoothed path between the reactant and product geometries.
    """
    reactant = copy_atoms(reactant)
    product = copy_atoms(product)

    # Read the initial geometries.
    chemical_symbols = reactant.get_chemical_symbols()

    # First redistribute number of images.
    # Perform interpolation if too few and subsampling if too many images are given
    raw_interpolated_positions = redistribute(
        chemical_symbols,
        [reactant.positions, product.positions],
        n_images,
        tol=redistribute_tol,
    )

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    geodesic_smoother = Geodesic(
        chemical_symbols,
        raw_interpolated_positions,
        morse_scaling,
        threshold=distance_cutoff,
        friction=geometry_friction,
        log_level=logging.DEBUG,
    )
    if perform_sweep == "auto":
        perform_sweep = len(chemical_symbols) > sweep_cutoff_size
    if perform_sweep:
        geodesic_smoother.sweep(
            tol=smoother_tol, max_iter=max_iterations, micro_iter=max_micro_iterations
        )
    else:
        geodesic_smoother.smooth(tol=smoother_tol, max_iter=max_iterations)
    atoms_list = [
        Atoms(symbols=chemical_symbols, positions=geom)
        for geom in geodesic_smoother.path
    ]
    return atoms_list


def run_geodesic_interpolate(
    pos_reactant: np.ndarray,
    pos_product: np.ndarray,
    z,
    calc,
    opt_kwargs={},
    return_middle_image=False,
):
    # initialize atoms objects
    atoms_r, _ = before_ase_opt(pos_reactant, z, calc=calc)
    atoms_p, _ = before_ase_opt(pos_product, z, calc=calc)

    _opt_kwargs = dict(
        n_images=3,
        # perform_sweep="auto",
        # redistribute_tol=1e-2,
        # smoother_tol=2e-3,
        # max_iterations=15,
        # max_micro_iterations=20,
        # morse_scaling=1.7,
        # geometry_friction=1e-2,
        # distance_cutoff=3.0,
        # sweep_cutoff_size=35,
    )
    _opt_kwargs.update(opt_kwargs)
    # run geodesic interpolation
    atoms_list = geodesic_interpolate_wrapper(
        reactant=atoms_r,
        product=atoms_p,
        **_opt_kwargs,
    )
    if return_middle_image:
        assert (
            _opt_kwargs["n_images"] % 2 == 1
        ), "n_images must be odd for return_middle_image"
        return atoms_list[math.floor(_opt_kwargs["n_images"] / 2)]
    else:
        return atoms_list
