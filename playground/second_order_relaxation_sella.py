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
from gadff.equiformer_torch_calculator import EquiformerTorchCalculator

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
from gadff.trajectorysaver import MyTrajectory
from gadff.equiformer_ase_calculator import EquiformerASECalculator

# https://github.com/virtualzx-nad/geodesic-interpolate
from geodesic_interpolate.geodesic import Geodesic
from geodesic_interpolate.interpolation import redistribute

###########################################################################
# Helper functions
###########################################################################


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


def get_hessian_function(hessian_method, asecalc):
    """Function that returns a (N*3, N*3) hessian matrix"""
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


###########################################################################
# Sella-specific functions
###########################################################################


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
    order,  # 1 = first-order saddle point, 0 = minimum
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

    # Create ASE Atoms object from the initial guess position
    start_pos = to_numpy(start_pos)
    atomic_numbers_np = to_numpy(z)
    mol_ase = Atoms(numbers=atomic_numbers_np, positions=start_pos)
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

    # Set up a Sella Dynamics object with improved parameters for TS search
    _sella_kwargs = dict(
        atoms=mol_ase,
        # constraints=cons,
        # trajectory=os.path.join(logfolder, "cu_sella_ts.traj"),
        trajectory=MyTrajectory(atoms=mol_ase),
        order=order,
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

    summary.update(
        {
            "sella_kwargs": _sella_kwargs,
            "run_kwargs": _run_kwargs,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

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

    """Computes RMSD between predicted and true transition state, and plots trajectory"""
    trajectory = summary.get("trajectory", None)
    rmsd_initial = summary.get("rmsd_initial", float("inf"))
    calcname = summary.get("calcname", "")
    if (trajectory is not None) and (true_pos is not None):
        pred_pos = trajectory[-1]
        # Compute RMSD between Sella-optimized structure and true transition state
        rmsd_final = align_ordered_and_get_rmsd(pred_pos, true_pos)
        print(f"RMSD end: {rmsd_final:.6f} Å")
        print(f"Improvement: {rmsd_initial - rmsd_final:.6f} Å")
        summary["rmsd_final"] = rmsd_final
        summary["rmsd_improvement"] = rmsd_initial - rmsd_final

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
    cleansummary = clean_dict(summary)
    with open(os.path.join(_this_plot_dir, "summary.json"), "w") as f:
        json.dump(cleansummary, f)
    print(f"Saved summary to {os.path.join(_this_plot_dir, 'summary.json')}")
    with open(os.path.join(_this_plot_dir, "summary.txt"), "w") as f:
        f.write(json.dumps(cleansummary, indent=4))
    print(f"Saved summary to {os.path.join(_this_plot_dir, 'summary.txt')}")

    return summary
