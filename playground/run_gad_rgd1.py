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
from recipes.gad import integrate_dynamics, run_sella, before_ase, after_ase
from recipes.quacc.equiformer.ts import ts_job, irc_job, quasi_irc_job, neb_job, geodesic_job

from quacc.atoms.ts import geodesic_interpolate_wrapper

this_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(this_dir, "plots_gad")
log_dir = os.path.join(this_dir, "logs_gad")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def test_gad_ts_search(sample, calc, eigen_method, x_lininter_rp):
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
        plot_dir=plot_dir,
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
        plot_dir=plot_dir,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_perturbed_ts"] = _rmsd_ts

    # Test run - start from reactant
    traj, _, _, _ = integrate_dynamics(
        sample.pos_reactant,
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
        plot_dir=plot_dir,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.1_s100"] = _rmsd_ts

    # Start from reactant
    traj, _, _, _ = integrate_dynamics(
        sample.pos_reactant,
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
        plot_dir=plot_dir,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.01_s1000"] = _rmsd_ts

    # large steps
    traj, _, _, _ = integrate_dynamics(
        sample.pos_reactant,
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
        plot_dir=plot_dir,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.1_s1000"] = _rmsd_ts

    # very long
    traj, _, _, _ = integrate_dynamics(
        sample.pos_reactant,
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
        plot_dir=plot_dir,
    )
    _rmsd_ts = align_ordered_and_get_rmsd(
        traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
    )
    results["ts_from_r_dt0.1_s10000"] = _rmsd_ts

    # Follow the GAD vector field from R-P interpolation
    traj, _, _, _ = integrate_dynamics(
        x_lininter_rp,
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
        plot_dir=plot_dir,
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

def main(
    eigen_method="qr",
    do_gad=False,
    do_sella=False,
    do_sella_quacc=False,
    do_sella_hessian=False,
    do_forces=False,
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

    # # Example forward pass
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
    x_lininter_rts = (1 - alpha) * pos_reactant + alpha * pos_transition
    plot_molecule_mpl(
        x_lininter_rts,
        atomic_numbers=sample.z,
        title="Initial guess R-TS interpolation",
        plot_dir=plot_dir,
        save=True,
    )

    x_lininter_rp = (1 - alpha) * pos_reactant + alpha * pos_product
    plot_molecule_mpl(
        x_lininter_rp,
        atomic_numbers=sample.z,
        title="Initial guess R-P interpolation",
        plot_dir=plot_dir,
        save=True,
    )

    ###################################################################################
    # Follow the GAD vector field to find the transition state
    
    if do_gad:
        test_gad_ts_search(sample, calc, eigen_method, x_lininter_rp)

    ###################################################################################
    if do_sella:
        print("=" * 60)
        print("Following Sella to find transition state")

        # See if Sella can find the transition state
        asecalc = EquiformerASECalculator(device=device)
        
        for hessian_method in [None, "autodiff", "predict"]:
            
            if hessian_method == "autodiff":
                def hessian_function(atoms): 
                    return asecalc.get_hessian_autodiff(atoms).reshape((3 * len(atoms), 3 * len(atoms)))
            elif hessian_method == "predict":
                def hessian_function(atoms):
                    return asecalc.get_hessian_prediction(atoms).reshape((3 * len(atoms), 3 * len(atoms)))
            elif hessian_method is None:
                hessian_function = None
            else:
                raise ValueError(f"Invalid method: {hessian_method}")

            # Linear interpolation between reactant and product
            mol_ase = run_sella(
                x_lininter_rp,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"TS from R-P | Hessian={hessian_method}",
                calc=asecalc,
                hessian_function=hessian_function,
            )

            # Linear interpolation between reactant and TS
            mol_ase = run_sella(
                x_lininter_rts,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"TS from R-TS | Hessian={hessian_method}",
                calc=asecalc,
                hessian_function=hessian_function,
            )

            # Start from reactant
            mol_ase = run_sella(
                pos_reactant,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"TS from R | Hessian={hessian_method}",
                calc=asecalc,
                hessian_function=hessian_function,
            )
        
        # # See if Sella can find the transition state with EMT calculator
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
    if do_sella_quacc:
        print("=" * 60)
        print("Quacc recipes")
        
        asecalc = EquiformerASECalculator(device=device)
        
        # build atoms object
        mol_ase, initsummary = before_ase(start_pos=x_lininter_rp, z=sample.z, true_pos=sample.pos_transition, calc=calc)
        # run function
        result = ts_job(
            atoms=mol_ase.copy(),
            use_custom_hessian=False,
            run_freq=True,
            freq_job_kwargs=None,
            opt_params=None,
            additional_fields=None,
            calc=calc,
        )
        result.update(initsummary)
        # eval and plot
        endsummary = after_ase(result, z=sample.z, title="TS QuAcc from R-P", true_pos=sample.pos_transition, plot_dir=plot_dir)
        result.update(endsummary)
        
        
        result = irc_job
        # build atoms object
        mol_ase, initsummary = before_ase(start_pos=sample.pos_transition, z=sample.z, true_pos=sample.pos_transition, calc=calc)
        # run function
        result = irc_job(
            atoms=mol_ase.copy(),
            direction= "forward", # forward or reverse
            run_freq=True,
            freq_job_kwargs=None,
            opt_params=None,
            additional_fields=None,
            calc=calc,
        )
        result.update(initsummary)
        # eval and plot
        endsummary = after_ase(result, z=sample.z, title="Forward IRC QuAcc from R", true_pos=sample.pos_transition, plot_dir=plot_dir)
        result.update(endsummary)
        
        
        result = neb_job(
            reactant_atoms=sample.pos_reactant,
            product_atoms=sample.pos_product,
            interpolation_method="linear", # "linear", "idpp" and "geodesic"
            relax_job_kwargs=None,
            interpolate_kwargs=None,
            neb_kwargs=None,
        )
        
        result = geodesic_job(
            reactant_atoms=sample.pos_reactant,
            product_atoms=sample.pos_product,
            relax_job_kwargs=None,
            geodesic_interpolate_kwargs=None,
        )
        
        atoms_list = geodesic_interpolate_wrapper(
            reactant=sample.pos_reactant,
            product=sample.pos_product,
            n_images=10, # MEP guess for NEB
            perform_sweep="auto",
            redistribute_tol=1e-2,
            smoother_tol=2e-3,
            max_iterations=15,
            max_micro_iterations=20,
            morse_scaling=1.7,
            geometry_friction=1e-2,
            distance_cutoff=3.0,
            sweep_cutoff_size=35,
        )
    

    ###################################################################################
    # Follow the forces to find the reactant minimum
    if do_forces:
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
            plot_dir=plot_dir,
        )
        plot_molecule_mpl(
            trajectory_pos[-1],
            atomic_numbers=sample.z,
            title="Optimized Minimum from R-TS interpolation",
            plot_dir=plot_dir,
            save=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAD-RGD1 main.")
    parser.add_argument(
        "--eigen-method",
        type=str,
        default="qr",
        help="Eigenvalue method for GAD (qr, svd, svdforce, inertia, geo, ase, eckartsvd, eckartqr)",
    )
    parser.add_argument(
        "--do-gad",
        action="store_true",
        help="Run GAD",
    )
    parser.add_argument(
        "--do-sella",
        action="store_true",
        help="Run Sella",
    )
    parser.add_argument(
        "--do-sella-quacc",
        action="store_true",
        help="Run Sella with QuAcc recipes",
    )
    parser.add_argument(
        "--do-sella-hessian",
        action="store_true",
        help="Run Sella with Equiformer Hessian",
    )
    parser.add_argument(
        "--do-forces",
        action="store_true",
        help="Run forces to find reactant minimum",
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

    results = main(
        eigen_method=args.eigen_method,
        do_gad=args.do_gad,
        do_sella=args.do_sella,
        do_sella_quacc=args.do_sella_quacc,
        do_sella_hessian=args.do_sella_hessian,
        do_forces=args.do_forces,
    )
    print("Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")
