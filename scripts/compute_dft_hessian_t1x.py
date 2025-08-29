from __future__ import annotations

import os
import argparse
from typing import List, Tuple

import numpy as np
import h5py
from tqdm import tqdm
import scipy.constants as spc

from pyscf import dft, gto


# Bohr radius in m
BOHR2M = spc.value("Bohr radius")
# Bohr -> Å conversion factor
BOHR2ANG = BOHR2M * 1e10
# Hartree to eV
AU2EV = spc.value("Hartree energy in eV")


def build_molecule(
    atoms_bohr: List[Tuple[int, Tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
) -> gto.Mole:
    """
    Build a PySCF molecule from a list like [(Z, (x,y,z)), ...] where coordinates are in Bohr.
    """
    spin = multiplicity - 1  # 2S = multiplicity - 1
    mol = gto.Mole()
    mol.atom = atoms_bohr  # list[(Z|(symbol), (x,y,z))]
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = "6-31g(d)"
    mol.unit = "Bohr"
    mol.build()
    return mol


def compute_hessian_au_bohr2(
    mol: gto.Mole, multiplicity: int = 1, xc: str = "wb97x", debug_hint: str = ""
) -> np.ndarray | None:
    """
    Compute DFT Hessian with PySCF. Returns Hessian in atomic units (Hartree/Bohr^2)
    with shape (3N, 3N). Returns None if SCF fails.
    """
    is_open_shell = multiplicity != 1
    if is_open_shell:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.verbose = 0
    try:
        mf.kernel()
    except Exception as e:
        print("\n" + ">" * 40)
        print(f"Error in SCF: {debug_hint}: \n{e}")
        print("<" * 40)
        return None
    if not mf.converged:
        print("\n" + ">" * 40)
        print(f"SCF did not converge: {debug_hint}")
        print("<" * 40)
        return None

    # (N, N, 3, 3) where N is number of atoms
    hessian = mf.Hessian().kernel()
    N = mol.natm
    hes = hessian.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    return hes


def au_bohr2_to_ev_ang2(hess_au_bohr2: np.ndarray) -> np.ndarray:
    """Convert Hessian from Hartree/Bohr^2 to eV/Å^2."""
    return hess_au_bohr2 * (AU2EV / (BOHR2ANG * BOHR2ANG))


def main():
    ap = argparse.ArgumentParser(description="Compute DFT Hessians for Transition1x val reactants")
    ap.add_argument(
        "--source_h5",
        type=str,
        default=os.path.abspath(os.path.join("Transition1x", "data", "transition1x.h5")),
        help="Path to the original Transition1x HDF5 file",
    )
    ap.add_argument(
        "--dest_h5",
        type=str,
        default=os.path.abspath(os.path.join("data", "t1x_val_reactant_hessian_100.h5")),
        help="Path to output HDF5 file with Hessians (eV/Å^2)",
    )
    ap.add_argument("--limit", type=int, default=100, help="Number of val reactants to process")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.dest_h5), exist_ok=True)

    with h5py.File(args.source_h5, "r") as src, h5py.File(args.dest_h5, "w") as dst:
        if "val" not in src:
            raise RuntimeError("'val' split not found in source HDF5")
        val_src = src["val"]
        val_dst = dst.create_group("val")

        count = 0
        val_index = 0  # index within original val split, by traversal order
        pbar = tqdm(total=args.limit, desc="Computing Hessians (val reactants)")
        for formula, grp in val_src.items():
            if formula not in val_dst:
                g_formula = val_dst.create_group(formula)
            else:
                g_formula = val_dst[formula]

            for rxn, subgrp in grp.items():
                if "reactant" not in subgrp:
                    continue

                # Copy full original group to preserve all keys and shapes
                dst.copy(subgrp, g_formula, name=rxn)

                g_rxn = g_formula[rxn]
                g_reactant = g_rxn["reactant"]

                reactant_grp = subgrp["reactant"]

                # Extract geometry for Hessian computation
                atomic_numbers = np.array(reactant_grp["atomic_numbers"], dtype=int)
                positions_all = np.array(reactant_grp["positions"])  # shape (T, N, 3) or (N, 3)

                positions = positions_all[0] if positions_all.ndim == 3 else positions_all

                atoms_bohr: List[Tuple[int, Tuple[float, float, float]]] = [
                    (int(Z), (float(x / BOHR2ANG), float(y / BOHR2ANG), float(z / BOHR2ANG)))
                    for Z, (x, y, z) in zip(atomic_numbers, positions)
                ]

                mol = build_molecule(atoms_bohr)
                hessian_au = compute_hessian_au_bohr2(
                    mol, multiplicity=1, xc="wb97x", debug_hint=f"{formula}/{rxn}"
                )
                if hessian_au is None:
                    # On failure, remove the copied reaction to avoid partial data
                    del g_formula[rxn]
                    val_index += 1
                    continue

                hessian_ev_ang2 = au_bohr2_to_ev_ang2(hessian_au)

                # New Hessian key in eV/Å^2 under reactant
                g_reactant.create_dataset(
                    "wB97x_6-31G(d).hessian", data=hessian_ev_ang2, compression="gzip"
                )
                # Store original val index for this reactant
                if "idx" in g_reactant:
                    del g_reactant["idx"]
                g_reactant.create_dataset("idx", data=np.array(val_index, dtype=np.int64))

                count += 1
                pbar.update(1)
                val_index += 1
                if count >= args.limit:
                    if "count" in dst:
                        del dst["count"]
                    dst.create_dataset("count", data=np.array(count, dtype=np.int64))
                    pbar.close()
                    return

        pbar.close()
        if "count" in dst:
            del dst["count"]
        dst.create_dataset("count", data=np.array(count, dtype=np.int64))


if __name__ == "__main__":
    main()


