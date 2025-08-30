from __future__ import annotations

import numpy as np
import argparse
import pandas as pd
import os
import glob
import shutil
from tqdm import tqdm
from typing import List, Tuple
import h5py
import scipy.constants as spc

from pyscf import dft, gto

# Bohr radius in m
BOHR2M = spc.value("Bohr radius")
# Bohr -> Å conversion factor
BOHR2ANG = BOHR2M * 1e10
# # Å -> Bohr conversion factor
# ANG2BOHR = 1 / BOHR2ANG
# Hartree to eV
AU2EV = spc.value("Hartree energy in eV")


def read_xyz(path: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Reads from XYZ file. Units are Angstrom.
    Returns a list of tuples (symbol, (x,y,z)).
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    try:
        natoms = int(lines[0].split()[0])
    except Exception as exc:
        raise ValueError(f"Invalid XYZ header in {path}: {exc}") from exc

    atom_lines = lines[1:]
    atoms: List[Tuple[str, Tuple[float, float, float]]] = []
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if i == 0:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except:
                # probably extra info like energy
                continue
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: '{ln}'")
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception as exc:
            raise ValueError(f"Invalid coordinates in line: '{ln}'") from exc
        atoms.append((sym, (x, y, z)))
    if len(atoms) != natoms:
        raise ValueError(
            f"XYZ atom count mismatch: header says {natoms}, parsed {len(atoms)} in {path}"
        )
    return atoms


def build_molecule(
    atoms: List[Tuple[str, Tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
) -> gto.Mole:
    """
    Build a PySCF molecule from a list of atoms [(symbol, (x,y,z)), (symbol, (x,y,z)), ...].
    Expects Bohr coordinates!
    """
    spin = multiplicity - 1  # 2S = multiplicity - 1
    mol = gto.Mole()
    mol.atom = atoms  # list[(symbol, (x,y,z))]
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = "6-31g(d)"
    mol.unit = "Bohr"
    mol.build()
    return mol


def compute_hessian(
    mol: gto.Mole, multiplicity: int = 1, xc: str = "wb97x", debug_hint=""
) -> np.ndarray:
    is_open_shell = multiplicity != 1
    if is_open_shell:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc
    # Tighten SCF a bit for stability
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.verbose = 0  # Suppress SCF convergence messages
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

    # Use the generic interface available on the SCF object
    # (N, N, 3, 3) where N is number of atoms
    hessian = mf.Hessian().kernel()

    # Properly reshape from (N, N, 3, 3) to (3N, 3N)
    # Need to transpose axes to get proper ordering: [atom_i, coord_i, atom_j, coord_j]
    N = mol.natm
    hes = hessian.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)

    # hes shape: (3N, 3N)
    # atomic units (Hartree/Bohr^2)
    return hes


def dft_hessian_from_xyz(xyz_path: str) -> None:
    """Compute DFT Hessian at ωB97X/6-31G(d) from an XYZ geometry using PySCF.
    Hessian is in atomic units (Hartree/Bohr^2).
    """
    xyz_path = os.path.abspath(xyz_path)
    atoms = read_xyz(xyz_path)  # Angstrom
    # convert Angstrom to Bohr
    atoms = [
        (sym, (x / BOHR2ANG, y / BOHR2ANG, z / BOHR2ANG)) for sym, (x, y, z) in atoms
    ]
    mol = build_molecule(atoms)
    hessian_dft = compute_hessian(mol, xc="wb97x")
    return hessian_dft


if __name__ == "__main__":
    """Compute DFT Hessian at ωB97X/6-31G(d) from an XYZ geometry using PySCF.
    Hessians are saved in Hartree/Bohr^2.
    """

    # loop over xyz files in data/relaxations
    xyz_files = glob.glob("data/relaxations/*.xyz")
    for xyz_file in tqdm(xyz_files):
        hessian_path = xyz_file.replace(".xyz", "_hessian.npy")
        # if <name>_hessian.npy exists, skip
        # else, compute and save
        if os.path.exists(hessian_path):
            continue
        hessian_dft = dft_hessian_from_xyz(xyz_file)
        np.save(hessian_path, hessian_dft)
