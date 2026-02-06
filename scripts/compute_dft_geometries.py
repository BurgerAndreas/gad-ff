"""
Compute DFT energy, forces, and Hessian for downloaded PubChem geometries
and save them in LMDB format compatible with the existing eval pipeline.

Uses PySCF with wB97X/6-31g(d), matching the level of theory used elsewhere
in this project. Will use GPU-accelerated PySCF (gpu4pyscf) if available.

Usage:
    uv run scripts/compute_dft_geometries.py
    uv run scripts/compute_dft_geometries.py --geom_csv geometries/geometries.csv --output dft_geometries.lmdb
    uv run scripts/compute_dft_geometries.py --max_samples_per_natoms 3
"""

import argparse
import json
import os
import pickle
import traceback

import lmdb
import numpy as np
import pandas as pd
import torch
from ase.io import read
from pyscf import gto
from gadff.constants import BOHR2ANG, AU2EV
from torch_geometric.data import Data
from tqdm import tqdm

from nets.prediction_utils import onehot_convert

SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8}

# Ground-state spin multiplicities (2S) for free atoms
ATOM_SPIN = {1: 1, 6: 2, 7: 3, 8: 2}

# Try to use GPU-accelerated PySCF
try:
    from gpu4pyscf.dft import rks as gpu_rks
    HAS_GPU4PYSCF = True
    print("Using gpu4pyscf (GPU-accelerated DFT)")
except ImportError:
    HAS_GPU4PYSCF = False
    print("gpu4pyscf not available, using CPU PySCF")
    print(traceback.format_exc())


def compute_atom_energies():
    """Compute wB97X/6-31g(d) free-atom energies (eV) for H, C, N, O.

    Uses UKS for open-shell atoms with the same grid settings as molecular
    calculations so atomization energies are consistent.
    """
    atom_energies = {}
    for z, spin in ATOM_SPIN.items():
        mol = gto.Mole()
        mol.atom = [(z, (0.0, 0.0, 0.0))]
        mol.charge = 0
        mol.spin = spin
        mol.basis = "6-31g(d)"
        mol.unit = "Bohr"
        mol.verbose = 0
        mol.build()

        if HAS_GPU4PYSCF:
            from gpu4pyscf.dft import uks as gpu_uks
            mf = gpu_uks.UKS(mol).density_fit()
        else:
            from pyscf import dft
            mf = dft.UKS(mol).density_fit()
        mf.xc = "wb97x"
        mf.verbose = 0
        mf.kernel()

        if not mf.converged:
            raise RuntimeError(f"SCF for free atom Z={z} did not converge")

        atom_energies[z] = mf.e_tot * AU2EV
        print(f"  Z={z:2d}  E={atom_energies[z]:.8f} eV")

    return atom_energies


def compute_dft(atomic_numbers, coords_ang):
    """
    Run wB97X/6-31g(d) single point: energy, gradient, Hessian.
    Uses GPU via gpu4pyscf if available, otherwise falls back to CPU.

    Returns dict with energy (eV), forces (eV/A), hessian (eV/A^2)
    or None if SCF fails to converge.
    """
    coords_bohr = coords_ang / BOHR2ANG

    atoms_bohr = [
        (int(z), (float(x), float(y), float(z_)))
        for z, (x, y, z_) in zip(atomic_numbers, coords_bohr)
    ]
    mol = gto.Mole()
    mol.atom = atoms_bohr
    mol.charge = 0
    mol.basis = "6-31g(d)"
    mol.unit = "Bohr"
    mol.verbose = 0

    total_electrons = sum(int(z) for z in atomic_numbers)
    if total_electrons % 2 == 0:
        mol.spin = 0
        mol.build()
        if HAS_GPU4PYSCF:
            mf = gpu_rks.RKS(mol).density_fit()
        else:
            from pyscf import dft
            mf = dft.RKS(mol).density_fit()
    else:
        mol.spin = 1
        mol.build()
        print(f"  Odd electrons ({total_electrons}), using UKS with spin=1")
        if HAS_GPU4PYSCF:
            from gpu4pyscf.dft import uks as gpu_uks
            mf = gpu_uks.UKS(mol).density_fit()
        else:
            from pyscf import dft
            mf = dft.UKS(mol).density_fit()
    mf.xc = "wb97x"
    mf.verbose = 0
    mf.kernel()

    if not mf.converged:
        return None

    energy_au = mf.e_tot  # Hartree

    # Gradient -> forces
    grad_au = mf.Gradients().kernel()  # (N, 3) in Hartree/Bohr
    forces_au = -grad_au  # (N, 3)

    # Hessian
    hobj = mf.Hessian()
    hessian_au = hobj.kernel()  # (N, N, 3, 3)
    N = mol.natm
    hess_cart_au = hessian_au.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)

    # Convert units: Hartree -> eV, Bohr -> Angstrom
    energy_ev = energy_au * AU2EV
    forces_ev_ang = forces_au * AU2EV / BOHR2ANG  # eV/Angstrom
    hess_ev_ang2 = hess_cart_au * AU2EV / (BOHR2ANG ** 2)  # eV/Angstrom^2

    return {
        "energy": energy_ev,
        "forces": forces_ev_ang,
        "hessian": hess_ev_ang2,
    }


def sdf_to_data(sdf_path, dft_result, atom_energies):
    """Convert SDF file + DFT results into a torch_geometric Data object."""
    atoms = read(sdf_path)
    coords = atoms.get_positions().astype(np.float32)  # (N, 3) Angstrom
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atomic_numbers)

    ae = dft_result["energy"] - sum(atom_energies[z] for z in atomic_numbers)

    device = torch.device("cpu")
    one_hot = onehot_convert(atomic_numbers.tolist(), device)

    data = Data(
        pos=torch.tensor(coords, dtype=torch.float32),
        z=torch.tensor(atomic_numbers, dtype=torch.int64),
        one_hot=one_hot,
        natoms=torch.tensor([n_atoms], dtype=torch.int64),
        energy=torch.tensor([dft_result["energy"]], dtype=torch.float64),
        ae=torch.tensor([ae], dtype=torch.float64),
        forces=torch.tensor(dft_result["forces"], dtype=torch.float32),
        hessian=torch.tensor(dft_result["hessian"], dtype=torch.float32),
    )
    return data


def write_lmdb(data_list, lmdb_path):
    """Write a list of Data objects to an LMDB file."""
    map_size = 1099511627776  # 1 TB
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False)

    with env.begin(write=True) as txn:
        for i, data in enumerate(data_list):
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data))
        txn.put("length".encode("ascii"), pickle.dumps(len(data_list)))

    env.close()


def load_checkpoint(ckpt_path):
    """Load checkpoint dict {sdf_path: {"status": ..., "result": ...}}."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_checkpoint(ckpt, ckpt_path):
    """Atomically save checkpoint by writing to a temp file then renaming."""
    tmp = ckpt_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Compute DFT for PubChem geometries")
    parser.add_argument(
        "--geom_csv",
        type=str,
        default="geometries/geometries.csv",
        help="Path to geometries CSV from fetch_geometries.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="geometries/dft_geometries.lmdb",
        help="Output LMDB path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max total geometries to process",
    )
    parser.add_argument(
        "--max_samples_per_natoms",
        type=int,
        default=1,
        help="Max geometries to process per unique natoms value",
    )
    args = parser.parse_args()

    print("Computing free-atom reference energies...")
    atom_energies = compute_atom_energies()
    print("atom_energies", atom_energies)
    print()

    df = pd.read_csv(args.geom_csv)
    df = df.sort_values("natoms").reset_index(drop=True)
    if args.max_samples_per_natoms is not None:
        df = df.groupby("natoms").head(args.max_samples_per_natoms).reset_index(drop=True)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    # Checkpoint file sits next to the output LMDB
    ckpt_path = args.output.replace(".lmdb", "_ckpt.pkl")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ckpt = load_checkpoint(ckpt_path)

    n_cached = sum(1 for v in ckpt.values() if v["status"] == "ok")
    print(f"Processing {len(df)} geometries from {args.geom_csv} ({n_cached} cached)")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="DFT"):
        sdf_path = row["file"]
        natoms = row["natoms"]
        print(f"sdf_path={sdf_path}  natoms={natoms}")

        # Skip if already computed
        if sdf_path in ckpt:
            print("Skipping because already computed")
            continue

        if not os.path.exists(sdf_path):
            print(f"  Missing: {sdf_path}")
            print("Skipping because missing")
            continue

        try:
            atoms = read(sdf_path)
            atomic_numbers = atoms.get_atomic_numbers()
            coords_ang = atoms.get_positions()

            # Check elements are H, C, N, O only
            allowed = {1, 6, 7, 8}
            if not set(atomic_numbers).issubset(allowed):
                print(f"  Skipping {sdf_path}: unsupported elements")
                ckpt[sdf_path] = {"status": "skipped", "result": None}
                save_checkpoint(ckpt, ckpt_path)
                continue

            result = compute_dft(atomic_numbers, coords_ang)
            if result is None:
                print(f"  SCF failed: {sdf_path}")
                ckpt[sdf_path] = {"status": "scf_failed", "result": None}
                save_checkpoint(ckpt, ckpt_path)
                continue

            ckpt[sdf_path] = {"status": "ok", "result": result}
            save_checkpoint(ckpt, ckpt_path)
            print(f"  OK: {sdf_path} | {len(atomic_numbers)} atoms | E={result['energy']:.4f} eV")

        except Exception as e:
            print(f"  Error on {sdf_path}: {e}")
            traceback.print_exc()
            ckpt[sdf_path] = {"status": f"error: {e}", "result": None}
            save_checkpoint(ckpt, ckpt_path)

    # Build LMDB and status CSV from checkpoint
    data_list = []
    records = []
    for _, row in df.iterrows():
        sdf_path = row["file"]
        entry = ckpt.get(sdf_path)
        if entry is None:
            continue
        if entry["status"] == "ok":
            data = sdf_to_data(sdf_path, entry["result"], atom_energies)
            data_list.append(data)
            records.append({
                **row.to_dict(),
                "status": "ok",
                "energy_ev": entry["result"]["energy"],
                "ae_ev": data.ae.item(),
                "lmdb_idx": len(data_list) - 1,
            })
        else:
            records.append({**row.to_dict(), "status": entry["status"]})

    if len(data_list) == 0:
        print("No successful DFT calculations. Nothing to save.")
        return

    write_lmdb(data_list, args.output)
    print(f"\nSaved {len(data_list)} samples to {args.output}")

    status_csv = args.output.replace(".lmdb", "_status.csv")
    pd.DataFrame(records).to_csv(status_csv, index=False)
    print(f"Saved status log to {status_csv}")


if __name__ == "__main__":
    main()
