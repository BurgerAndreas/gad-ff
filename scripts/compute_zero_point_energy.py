"""
Zero-point energy (ZPE) for a molecular geometry is computed from the vibrational frequencies of the molecule at that geometry.

The steps are:
a. Optimize the molecular geometry (find the equilibrium structure).
b. Compute the vibrational frequencies by evaluating the Hessian (second derivative of energy with respect to nuclear displacements).
c. The zero-point energy is the sum over all vibrational mode with frequency v_i and energy (1/2) h v_i
ZPE = (1/2) Σ h v_i


In practice, we:
1. Start from T1x val reactants.
2. Relax the geometry further using BFGS with EquiformerV2
(if necessary we will have to use DFT via PySCF /ssd/Code/ReactBench/dependencies/pysisyphus/pysisyphus/calculators/PySCF.py )
3. Confirm we converged to max_force_dft < 10^-2 eV/Angstrom, ideally 10^-3 eV/Angstrom
4. Save the relaxed geometry
5. Compute the Hessian using DFT via PySCF, save the Hessian
6. Compute the Hessian using various models (AlphaNet/LEFT/LEFT-DF autograd, EquiformerV2 predict/autograd, see scripts/eval_horm.py)
7. Mass-weight and Eckart-project the Hessians for each method
8. Compute the ZPE for each method
9. Save ZPEs in dataframe and to csv
"""

import argparse
import os
import time
import pathlib
import numpy as np
import torch

# pysisyphus + Equiformer
from pysisyphus.Geometry import Geometry
from pysisyphus.optimizers.BFGS import BFGS
from pysisyphus.optimizers.RFOptimizer import RFOptimizer
from pysisyphus.constants import BOHR2ANG, AU2EV
from ReactBench.Calculators.equiformer import PysisEquiformer
from pysisyphus.calculators.PySCF import PySCF as PysisPySCF

# dataset + project paths
from gadff.t1x_dft_dataloader import Dataloader as T1xDFTDataloader
from gadff.path_config import ROOT_DIR

from gadff.frequency_analysis import analyze_frequencies, eckart_projection_notmw  # noqa: F401
from gadff.frequency_analysis import eigval_to_wavenumber

from alphanet.models.alphanet import AlphaNet
from leftnet.model.leftnet import LEFTNet

# DFT (PySCF)
from pyscf import dft, gto
import scipy.constants as spc
import pandas as pd

from gadff.horm.training_module import PotentialModule
from ocpmodels.common.relaxation.ase_utils import (
    coord_atoms_to_torch_geometric_hessian,
)
from nets.prediction_utils import compute_extra_props


Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


def onehot_convert(atomic_numbers):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder = {
        1: [1, 0, 0, 0, 0],
        6: [0, 1, 0, 0, 0],
        7: [0, 0, 1, 0, 0],
        8: [0, 0, 0, 1, 0],
    }
    onehot = [encoder[i] for i in atomic_numbers]
    return np.array(onehot)


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    """Helper function to compute derivatives"""
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    """Compute Hessian matrix using autograd."""
    # Compute forces if not given
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)

    # Get number of components (n_atoms * 3)
    n_comp = forces.reshape(-1).shape[0]

    # Initialize hessian
    hess = []
    for f in forces.reshape(-1):
        # Compute second-order derivative for each element
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)

    # Stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def _write_xyz(path, atomssymbols, coords_ang):
    n = len(atomssymbols)
    lines = [str(n), "relaxed geometry"]
    for el, (x, y, z) in zip(atomssymbols, coords_ang):
        lines.append(f"{el} {x:.8f} {y:.8f} {z:.8f}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def _read_xyz(path):
    lines = pathlib.Path(path).read_text().strip().splitlines()
    try:
        n = int(lines[0].strip())
    except Exception:
        raise ValueError("XYZ: first line must be atom count")
    body = lines[2 : 2 + n]
    atoms, coords3d = [], []
    for ln in body:
        el, x, y, z = ln.split()[:4]
        atoms.append(el)
        coords3d.append([float(x), float(y), float(z)])
    return atoms, np.asarray(coords3d, float)


def relax_t1x_reactants(
    dataset_path,
    out_dir,
    *,
    coord="redund",
    max_samples=50,
    max_cycles=200,
    thresh="gau",
    ckpt_path=None,
    device="cuda",
    verbose=False,
    force_fail_thresh_eV_A=None,
    ckpt_alpha="ckpt/alpha.ckpt",
    ckpt_left="ckpt/left.ckpt",
    ckpt_left_df="ckpt/left-df.ckpt",
    ckpt_eqv2_autograd="ckpt/eqv2.ckpt",
    redo_relax=False,
    redo_dft=False,
    dft_relax=False,
):
    os.makedirs(out_dir, exist_ok=True)
    xyz_dir = os.path.join(out_dir, "relaxed_xyz")
    os.makedirs(xyz_dir, exist_ok=True)
    dft_dir = os.path.join(out_dir, "dft")
    dft_grad_dir = os.path.join(dft_dir, "gradients")
    dft_hess_dir = os.path.join(dft_dir, "hessians")
    os.makedirs(dft_grad_dir, exist_ok=True)
    os.makedirs(dft_hess_dir, exist_ok=True)
    # No need to save model Hessians per user request

    # Default checkpoint (same as used elsewhere in this repo)
    if ckpt_path is None:
        ckpt_path = "/ssd/Code/ReactBench/ckpt/hesspred/hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746.ckpt"

    # Load dataset (validation reactants)
    dataset = T1xDFTDataloader(dataset_path, datasplit="val", only_final=True)

    # Calculator for relaxation: either Equiformer (predict) or PySCF DFT
    if not dft_relax:
        base_calc = PysisEquiformer(
            charge=0,
            ckpt_path=ckpt_path,
            config_path="auto",
            device=device,
            hessianmethod_name="predict",
            hessian_method="predict",
            mem=4000,
            method="equiformer",
            mult=1,
            pal=1,
        )
    else:
        base_calc = PysisPySCF(
            basis="6-31g(d)",
            xc="wb97x",
            method="dft",
            charge=0,
            mult=1,
            mem=4000,
            pal=1,
            verbose=0,
        )
    print(f"Initialized basecalc")

    # Load autograd models once (Alpha, LEFT, LEFT-DF, EquiformerV2 autograd)
    def _load_model(ckpt):
        m = PotentialModule.load_from_checkpoint(ckpt, strict=False).potential
        return m.to(device).eval()

    model_alpha = _load_model(ckpt_alpha)
    model_left = _load_model(ckpt_left)
    model_left_df = _load_model(ckpt_left_df)
    model_eqv2_predict = _load_model(ckpt_path)
    model_eqv2_autograd = _load_model(ckpt_eqv2_autograd)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    started = time.perf_counter()
    cnt_done = 0
    summary = []
    zpe_rows = []
    for idx, molecule in enumerate(dataset):
        if cnt_done >= max_samples:
            break
        print("=" * 80)
        print(f"Processing {cnt_done} | sample {idx}")
        print("=" * 80)

        reactant = molecule["reactant"]
        coords_ang = np.asarray(reactant["positions"], dtype=float)
        atomic_numbers = np.asarray(reactant["atomic_numbers"], dtype=int)
        atomssymbols = [Z_TO_SYMBOL.get(int(z), "X") for z in atomic_numbers]

        # Optimize with BFGS
        out_dir_sample = os.path.join(out_dir, f"sample_{idx:05d}")
        xyz_path = os.path.join(xyz_dir, f"reactant_{idx:05d}.xyz")
        if (
            not os.path.isdir(out_dir_sample)
            or redo_relax
            or (not os.path.isfile(xyz_path))
        ):
            os.makedirs(out_dir_sample, exist_ok=True)
            # Convert to Bohr for pysisyphus
            coords_bohr = coords_ang / BOHR2ANG

            # Build geometry + calculator
            geom = Geometry(atomssymbols, coords_bohr, coord_type=coord)
            geom.set_calculator(base_calc)
            # RFO setup: learned Hessian init (Equiformer) or unit init (DFT)
            opt = RFOptimizer(
                geom,
                thresh=thresh,
                trust_radius=0.3,
                hessian_init=("unit" if dft_relax else "calc"),
                hessian_update="bfgs",
                hessian_recalc=None,
                line_search=True,
                out_dir=out_dir_sample,
                max_cycles=max_cycles,
            )
            opt.run()

            # Save relaxed structure as XYZ (Angstrom)
            final_coords_ang = (geom._coords).reshape(-1, 3) * BOHR2ANG
            _write_xyz(xyz_path, atomssymbols, final_coords_ang)
        else:
            # load relaxed geometry
            atoms_xyz, coords_xyz = _read_xyz(xyz_path)
            # simple sanity: ensure same atom order
            if len(atoms_xyz) != len(atomssymbols) or any(
                a != b for a, b in zip(atoms_xyz, atomssymbols)
            ):
                raise ValueError(f"Atom symbols mismatch in {xyz_path}")
            final_coords_ang = coords_xyz
            coords_bohr = final_coords_ang / BOHR2ANG
            geom = Geometry(atomssymbols, coords_bohr.reshape(-1), coord_type=coord)
            geom.set_calculator(base_calc)

        hessian_path = os.path.join(
            dft_hess_dir, f"reactant_{idx:05d}.hessian_eV_A2.npy"
        )
        force_path = os.path.join(dft_grad_dir, f"reactant_{idx:05d}.forces_eV_A.npy")
        if (
            (not os.path.isfile(hessian_path))
            or (not os.path.isfile(force_path))
            or redo_dft
        ):
            print(f"\n# {cnt_done} running DFT for sample {idx}")
            # DFT at relaxed geometry (forces and Hessian)
            atoms_bohr = [
                (int(Z), (float(x), float(y), float(z)))
                for Z, (x, y, z) in zip(atomic_numbers, geom._coords.reshape(-1, 3))
            ]
            mol = gto.Mole()
            mol.atom = atoms_bohr
            mol.charge = 0
            mol.spin = 0
            mol.basis = "6-31g(d)"
            mol.unit = "Bohr"
            mol.build()

            mf = dft.RKS(mol)
            mf.xc = "wb97x"
            mf.conv_tol = 1e-12
            mf.max_cycle = 200
            mf.verbose = 0
            mf.grids.atom_grid = (99, 590)
            mf.grids.prune = None
            mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"PySCF SCF did not converge for sample {idx}")

            grad_au_bohr = mf.nuc_grad_method().kernel()  # (N,3)
            forces_au_bohr = -grad_au_bohr
            forces_eV_A = forces_au_bohr * (AU2EV / BOHR2ANG)
            max_force = float(np.max(np.linalg.norm(forces_eV_A, axis=1)))
            print(f" Max DFT force {max_force:.1e} eV/Å")
            if force_fail_thresh_eV_A is not None and max_force > float(
                force_fail_thresh_eV_A
            ):
                print(
                    f"Max DFT force {max_force:.4e} eV/Å exceeds threshold {force_fail_thresh_eV_A:.4e} eV/Å for sample {idx}"
                )
                continue

            hobj = mf.Hessian()
            setattr(hobj, "conv_tol", 1e-10)
            setattr(hobj, "max_cycle", 100)
            hessian_au = hobj.kernel()  # (N,N,3,3)
            N = mol.natm
            hess_cart_au = hessian_au.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
            hess_ev_ang2 = hess_cart_au * (AU2EV / (BOHR2ANG * BOHR2ANG))

            np.save(hessian_path, hess_ev_ang2)
            np.save(force_path, forces_eV_A)

        else:
            # load hessian and forces
            hess_ev_ang2 = np.load(hessian_path)
            forces_eV_A = np.load(force_path)
            max_force = float(np.max(np.linalg.norm(forces_eV_A, axis=1)))
            # Convert to atomic units for downstream
            hess_cart_au = hess_ev_ang2 / AU2EV * (BOHR2ANG * BOHR2ANG)

        summary.append(
            {
                "idx": int(idx),
                "natoms": int(len(atomssymbols)),
                "max_force_eV_A": max_force,
            }
        )

        # ----- Model Hessians (Equiformer) -----
        print(f"\n# {cnt_done} computing model Hessians for sample {idx}")

        # ---------- Compute model autograd Hessians at relaxed geometry (no saving) ----------
        def _hessian_autograd_with_model(
            model, _coords, _atomic_numbers, do_autograd=True
        ):
            # Build TG batch from coords (Angstrom) & atomic numbers
            use_pbc = getattr(model, "use_pbc", False)
            batch = coord_atoms_to_torch_geometric_hessian(
                coords=_coords,
                atomic_nums=_atomic_numbers,
                cutoff=getattr(model, "cutoff", 12.0),
                max_neighbors=getattr(model, "max_neighbors", None),
                use_pbc=use_pbc,
                with_grad=True,
                cutoff_hessian=getattr(model, "cutoff_hessian", 100.0),
            )
            # batch.one_hot = torch.tensor(onehot_convert(_atomic_numbers), dtype=torch.int64)
            batch.ae = torch.tensor(np.array([len(_atomic_numbers)]), dtype=torch.int64)
            batch = batch.to(device)
            batch = compute_extra_props(batch, pos_require_grad=True)

            # Forward passes per model type (mirrors eval_horm.py)
            model_name = model.__class__.__name__.lower()
            n_atoms = batch.pos.shape[0]
            if "leftnet" in model_name:
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward_autograd(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)
            elif "equiformer" in model_name:
                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=False
                    )
                    hessian_model = compute_hessian(
                        batch.pos, energy_model, force_model
                    )
                else:
                    with torch.no_grad():
                        energy_model, force_model, out = model.forward(
                            batch, otf_graph=False, hessian=True, add_props=True
                        )
                    hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)
            else:
                # AlphaNet (default)
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)

            hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)
            return hessian_model.detach().cpu().numpy()

        h_alpha_ev = _hessian_autograd_with_model(
            model_alpha, final_coords_ang.copy(), atomic_numbers.copy()
        )
        h_left_ev = _hessian_autograd_with_model(
            model_left, final_coords_ang.copy(), atomic_numbers.copy()
        )
        h_leftdf_ev = _hessian_autograd_with_model(
            model_left_df, final_coords_ang.copy(), atomic_numbers.copy()
        )

        # EquiformerV2
        h_eqv2_auto_ev = _hessian_autograd_with_model(
            model_eqv2_autograd, final_coords_ang.copy(), atomic_numbers.copy()
        )
        h_eqv2_pred_ev = _hessian_autograd_with_model(
            model_eqv2_predict,
            final_coords_ang.copy(),
            atomic_numbers.copy(),
            do_autograd=False,
        )

        # ---------- Compute ZPEs (Eckart + mass-weighting) for each method ----------
        print(f"\n{cnt_done} computing ZPEs for sample {idx}")

        def _to_au(h_ev_ang2):
            return h_ev_ang2 / AU2EV * (BOHR2ANG * BOHR2ANG)

        def _zpe_from_hessian_au(h_au, coords_bohr, atomsymbols):
            # Project TR modes on mass-weighted Hessian
            proj = eckart_projection_notmw(h_au, coords_bohr.reshape(-1), atomsymbols)
            eigvals, _ = np.linalg.eigh(proj)
            # Enforce minima: all vibrational eigenvalues must be positive
            allpos = True
            if not np.all(eigvals > 0):
                print(
                    f"Non-positive eigenvalues in projected Hessian: min={eigvals.min():.3e}"
                )
                allpos = False
            eigvals = eigvals[eigvals > 0]
            # Convert eigenvalues to wavenumbers (cm^-1)
            wavenumbers_cm = eigval_to_wavenumber(eigvals)
            # ZPE (J per molecule): 0.5 * h * c * sum(nu_bar[m^-1])
            zpe_J = 0.5 * spc.h * spc.c * np.sum(wavenumbers_cm * 100.0)
            # Convert to eV per molecule
            return zpe_J / spc.e

        # Prepare method Hessians in au
        h_dft_au = hess_cart_au
        h_alpha_au = _to_au(h_alpha_ev)
        h_left_au = _to_au(h_left_ev)
        h_leftdf_au = _to_au(h_leftdf_ev)
        h_eqv2_auto_au = _to_au(h_eqv2_auto_ev)
        h_eqv2_pred_au = _to_au(h_eqv2_pred_ev)

        zpe_rows.extend(
            [
                {
                    "idx": int(idx),
                    "model": "DFT",
                    "method": "DFT",
                    "zpe_eV": _zpe_from_hessian_au(
                        h_dft_au, geom._coords, atomssymbols
                    ),
                },
                {
                    "idx": int(idx),
                    "model": "AlphaNet",
                    "method": "autograd",
                    "zpe_eV": _zpe_from_hessian_au(
                        h_alpha_au, geom._coords, atomssymbols
                    ),
                },
                {
                    "idx": int(idx),
                    "model": "LeftNet",
                    "method": "autograd",
                    "zpe_eV": _zpe_from_hessian_au(
                        h_left_au, geom._coords, atomssymbols
                    ),
                },
                {
                    "idx": int(idx),
                    "model": "LeftNet-DF",
                    "method": "autograd",
                    "zpe_eV": _zpe_from_hessian_au(
                        h_leftdf_au, geom._coords, atomssymbols
                    ),
                },
                {
                    "idx": int(idx),
                    "model": "EquiformerV2",
                    "method": "predict",
                    "zpe_eV": _zpe_from_hessian_au(
                        h_eqv2_pred_au, geom._coords, atomssymbols
                    ),
                },
                {
                    "idx": int(idx),
                    "model": "EquiformerV2",
                    "method": "autograd",
                    "zpe_eV": _zpe_from_hessian_au(
                        h_eqv2_auto_au, geom._coords, atomssymbols
                    ),
                },
            ]
        )

        cnt_done += 1
        print(f"\n[{cnt_done}] saved {xyz_path}")

    elapsed = time.perf_counter() - started
    print(f"Relaxed {cnt_done} geometries in {elapsed:.2f}s. Output: {xyz_dir}")
    if len(summary) > 0:
        summ_path = os.path.join(out_dir, "dft", "summary.csv")
        os.makedirs(os.path.dirname(summ_path), exist_ok=True)
        with open(summ_path, "w") as fh:
            fh.write("idx,natoms,max_force_eV_A\n")
            for row in summary:
                fh.write(f"{row['idx']},{row['natoms']},{row['max_force_eV_A']:.8f}\n")
        print(f"Saved DFT summary to {summ_path}")

    # Save ZPEs to CSV
    if len(zpe_rows) > 0:
        zpe_dir = os.path.join(out_dir, "zpe")
        os.makedirs(zpe_dir, exist_ok=True)
        df = pd.DataFrame(zpe_rows)
        csv_path = os.path.join(zpe_dir, "zpe_all_methods.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved ZPEs to {csv_path}")

    return df


def print_zpe_latex_table(csv_path, ref_method="DFT", decimals=3):
    """Load the saved ZPE CSV and print a LaTeX table with specific columns/rows.

    Columns: "Hessian (autograd/predict)", "Model", "ZPE error".
    Rows (in order):
      - autograd AlphaNet
      - autograd LeftNet
      - autograd LeftNet-DF
      - autograd EquiformerV2
      - predict EquiformerV2

    ZPE error is formatted as: MAE (STD), where errors are (pred - ref).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"idx", "method", "zpe_eV"}
    missing = required - set(df.columns)
    if len(missing) > 0:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    ref_df = df[df["method"] == ref_method][["idx", "zpe_eV"]].rename(
        columns={"zpe_eV": "zpe_ref"}
    )
    if len(ref_df) == 0:
        raise ValueError(f"Reference method '{ref_method}' not present in CSV")

    merged = df.merge(ref_df, on="idx", how="inner")
    merged = merged[merged["method"] != ref_method].copy()
    merged["error"] = merged["zpe_eV"] - merged["zpe_ref"]
    merged["abs_error"] = merged["error"].abs()

    # Desired display rows using new DF format (separate columns: method, model)
    row_specs = [
        ("autograd", "AlphaNet", "autograd"),
        ("autograd", "LeftNet", "autograd"),
        ("autograd", "LeftNet-DF", "autograd"),
        ("autograd", "EquiformerV2", "autograd"),
        ("predict", "EquiformerV2", "predict"),  # may be absent
    ]

    rows = []
    for hess_kind, model_name, method_value in row_specs:
        sub = merged[(merged["method"] == method_value) & (merged["model"] == model_name)]
        if len(sub) == 0:
            zpe_err_str = "-"
        else:
            err = sub["error"].to_numpy()
            ae = np.abs(err)
            mae = float(ae.mean()) if len(ae) > 0 else np.nan
            std = float(err.std(ddof=0)) if len(err) > 0 else np.nan
            if np.isnan(mae) or np.isnan(std):
                zpe_err_str = "-"
            else:
                zpe_err_str = f"{mae:.{int(decimals)}f} ({std:.{int(decimals)}f})"
        rows.append(
            {
                "Hessian (autograd/predict)": hess_kind,
                "Model": model_name,
                "ZPE error": zpe_err_str,
            }
        )

    table = pd.DataFrame(
        rows, columns=["Hessian (autograd/predict)", "Model", "ZPE error"]
    )

    latex = table.to_latex(index=False, escape=True)
    print(latex)
    return latex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="data/t1x_val_reactant_hessian_100.h5",
        help="Path to T1x validation HDF5 file (reactant set)",
    )
    ap.add_argument(
        "--coord",
        type=str,
        default="redund",
        choices=["cart", "redund", "dlc", "tric"],
        help="Coordinate system",
    )
    ap.add_argument("--max_samples", type=int, default=10)
    ap.add_argument("--max_cycles", type=int, default=200)
    ap.add_argument("--thresh", type=str, default="gau")
    ap.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to Equiformer checkpoint"
    )
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--redo", type=bool, default=False)
    ap.add_argument("--redo_relax", type=bool, default=False)
    ap.add_argument("--redo_dft", type=bool, default=False)
    ap.add_argument(
        "--dft",
        type=bool,
        default=False,
        help="Relax using PySCF DFT with RFO+BFGS unit-init instead of Equiformer",
    )
    ap.add_argument("--verbose", type=bool, default=False)
    ap.add_argument(
        "--force_fail_thresh_eV_A",
        type=float,
        default=None,
        help="If set, raise when max force exceeds this (eV/Å)",
    )
    args = ap.parse_args()

    # Resolve output directory
    if args.out_dir is None:
        source_label = os.path.splitext(os.path.basename(args.dataset))[0]
        args.out_dir = os.path.join(
            ROOT_DIR,
            "runs_zpe",
            f"{source_label}_{args.coord}_{args.thresh.replace('_', '')}_{args.max_samples}",
            "dft" if args.dft else "equ",
        )
    if args.redo:
        # minimal cleanup: remove only XYZ dir
        xyz_dir = os.path.join(args.out_dir, "relaxed_xyz")
        if os.path.isdir(xyz_dir):
            for f in os.listdir(xyz_dir):
                p = os.path.join(xyz_dir, f)
                if os.path.isfile(p):
                    os.remove(p)

    relax_t1x_reactants(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        coord=args.coord,
        max_samples=args.max_samples,
        max_cycles=args.max_cycles,
        thresh=args.thresh,
        ckpt_path=args.ckpt_path,
        device=args.device,
        verbose=args.verbose,
        force_fail_thresh_eV_A=args.force_fail_thresh_eV_A,
        redo_relax=args.redo_relax,
        redo_dft=args.redo_dft,
        dft_relax=args.dft,
    )
    print_zpe_latex_table(os.path.join(args.out_dir, "zpe", "zpe_all_methods.csv"))


if __name__ == "__main__":
    main()
