import argparse
import time
import sys
import pathlib
import numpy as np
import os
import glob

from pysisyphus.Geometry import Geometry  # Geometry API + coordinate systems
from pysisyphus.calculators.LennardJones import (
    LennardJones,
)  # built-in pure-Python calculator
from pysisyphus.calculators.MLFF import MLFF
from pysisyphus.calculators.Calculator import Calculator  # base class to wrap/override
from pysisyphus.optimizers.FIRE import FIRE  # first-order baseline
from pysisyphus.optimizers.RFOptimizer import RFOptimizer  # second-order RFO + BFGS
from pysisyphus.optimizers.BFGS import BFGS

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path
from pytorch_geometric.data import DataLoader as TGDataLoader

import numpy as np
import h5py

from pysisyphus.helpers_pure import eigval_to_wavenumber
from pysisyphus.helpers import _do_hessian
from pysisyphus.io.hessian import save_hessian
from pysisyphus.constants import ANG2BOHR, AU2KJPERMOL

"""
- first-order (FIRE) optimizer for the baseline
- second-order RFO with BFGS updates, where we can choose the initial Hessian and tell it when to recompute the Hessian 
(we will supply your predicted Hessian through the calculator) ([pysisyphus.readthedocs.io][1])

run as:
python scripts/second_order_relaxation_pysiyphus.py molecule.xyz --coord redund

It implements four variants:

1. baseline: first-order (FIRE)
2. no-Hessian: RFO+BFGS with unit initial Hessian
3. initial-only: RFO+BFGS with Hpred only at step 0
4. periodic replace: RFO+BFGS with Hpred every k∈{3,1}

Notes
- The script reads a single-geometry XYZ (Å). pysisyphus supports XYZ directly and Geometry can be built from atoms + coords; use coord_type='cart' or 'redund' as you like. ([pysisyphus.readthedocs.io][2])
- For a “built-in calculator,” I wrapped the included Lennard-Jones calculator (pure Python) and overrode get_hessian to return your predicted Hessian. Swap in your predictor at predict_hessian(). Pure-Python calculators are documented here. ([pysisyphus.readthedocs.io][3])
- Metrics collected: gradient evaluations (counted at the calculator), wall time, steps (cycles), success flag. Optional trust-region diagnostics vary across versions; if you need them we can also parse the HDF5 dump that pysisyphus writes. ([pysisyphus.readthedocs.io][1])

What this is doing:
- baseline: FIRE -> first-order only, no Hessian
- no-Hessian: RFOptimizer(hessian_init='unit', hessian_update='bfgs') -> quasi-Newton with a diagonal initial guess (no external Hessian) ([pysisyphus.readthedocs.io][1])
- initial-only: RFOptimizer(hessian_init='calc', hessian_recalc=None) -> your H at step 0, then BFGS updates only ([pysisyphus.readthedocs.io][1])
- periodic replace: RFOptimizer(hessian_init='calc', hessian_recalc=k) -> your H injected every k steps (k=3,1) ([pysisyphus.readthedocs.io][1])

Why this wiring is idiomatic pysisyphus
- You can supply a Hessian via the calculator's get_hessian(); setting hessian_init='calc' makes the optimizer ask the calculator for H at step 0, and hessian_recalc=k repeats that every k cycles. We exploit that by letting HpredCalc return your predicted DFT-level Hessian. ([pysisyphus.readthedocs.io][1])
- The built-in pure-Python calculators (Lennard-Jones, TIP3P, etc.) let you test the plumbing with no external codes. Swap to XTB/Psi4 when you move from a toy potential to real molecules. ([pysisyphus.readthedocs.io][3])
- Coordinate systems: use redund (RIC) or dlc/tric for stability; pysisyphus supports all of them for geometry optimization. ([pysisyphus.readthedocs.io][2])

If you want the optional diagnostics
- Step rejection rate and trust-radius statistics can be pulled from the optimizer's dump (HDF5) if you instantiate RFOptimizer/FIRE with dump=True and parse optimization.h5 afterwards. pysisyphus exposes trust-region controls, and the docs list the trust-radius options. ([pysisyphus.readthedocs.io][1])

Caveats on units
Your predictor should return the Cartesian Hessian in the units consistent with your calculator (typically Hartree per distance squared). The wrapper calls prepare_coords to pass Å into your predictor; if your predictor expects Bohr, convert inside predict_hessian() to maintain consistency. The Hessian is then transformed into internals by pysisyphus under the hood. ([pysisyphus.readthedocs.io][2])

Docs I matched the code to
- Minimization options and RFO/BFGS/Hessian refresh knobs (hessian_init, hessian_update, hessian_recalc; trust-radius): pysisyphus “Minimization” page. ([pysisyphus.readthedocs.io][1])
- Geometry and coordinate systems; reading XYZ; internal vs Cartesian/DLC/TRIC: pysisyphus “Coordinate Systems”. ([pysisyphus.readthedocs.io][2])
- Built-in calculators, including pure-Python Lennard-Jones: pysisyphus “Calculators”. ([pysisyphus.readthedocs.io][3])

[1]: https://pysisyphus.readthedocs.io/en/latest/min_optimization.html "7. Minimization — pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"
[2]: https://pysisyphus.readthedocs.io/en/latest/coordinate_systems.html "5. Coordinate Systems — pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"
[3]: https://pysisyphus.readthedocs.io/en/latest/calculators.html "6. Calculators — pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"

Input parameters:

opt:
 type: rfo                      # Optimization algorithm
 max_cycles: 50                 # Maximum number of optimization cycles

 overachieve_factor: 2          # Indicate convergence, regardless of the
                                # proposed step when max(grad) and rms(grad)
                                # are overachieved by factor [n]

 do_hess: True                  # Calculate the hessian at the final geometry
                                # after the optimization.

 #hessian_recalc: None          # Recalculate exact hessian every n-th cylce

 #hessian_recalc_adapt: None    # Expects a float. Recalculate exact hessian
                                # whenever the gradient norms drops below
                                # 1/[n] of the gradient norm at the last hessian
                                # recalculation.

 #hessian_init: fischer         # Type of model hessian. Other options are: 'calc,
                                # simple, xtb, lindh, swart, unit'

 #hessian_update: bfgs          # Hessian-update. Other options are: 'flowchart,
                                # damped_bfgs, bofill'. bofill is not recommended
                                # for minimum searches.
 #small_eigval_thresh: 1e-8     # Neglect eigenvalues and corresponding eigenvectors
                                # below this threshold.

 #max_micro_cycles: 50          # No. of micro cycles for the RS-variants. Does not apply
                                # to TRIM.

 #trust_radius: 0.3             # Initial trust radius.
 #trust_max: 1.0                # Max. trust radius
 #trust_min: 0.1                # Min. trust radius

 #line_search: True             # Do line search

 #gdiis_thresh: 0.0025          # May do GDIIS if rms(step) falls below this threshold
 #gediis_thresh: 0.01           # May do GEDIIS if rms(grad) falls below this threshold
 #gdiis: True                   # Do controlled GDIIS after 'gdiis_thresh' is reached
 #gediis: False                 # Do GEDIIS after 'gediis_thresh' is reached

calc:
 type: turbomole
 control_path: control_path_pbe0_def2svp_s1     # Path to the prepared calculation
 track: True                                    # Activate excited state tracking
 ovlp_type: tden                                # Track with transition density matrix overlaps
 charge: 0
 mult: 1
 pal: 4
 mem: 2000

geom:
 type: redund
 fn: cytosin.xyz

"""

# we probably do not need this
# if hessian_method == "predict":
#     transform = HessianGraphTransform(
#         cutoff=model.cutoff,
#         max_neighbors=model.max_neighbors,
#         use_pbc=model.use_pbc,
#     )
# else:
#     transform = None

# dataset = LmdbDataset(fix_dataset_path(lmdb_path), transform=transform)
# dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

# --------------------------
#  Utilities
# --------------------------


def load_xyz(fn):
    """Read a single-geometry XYZ (Angstrom). Returns atoms (list[str]), coords (1d array, 3N)."""
    lines = pathlib.Path(fn).read_text().strip().splitlines()
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
    coords = np.asarray(coords3d, float).reshape(-1)  # 3N (Å)
    return atoms, coords


# --------------------------
#  Plug your predictor here
# --------------------------


def predict_hessian(atoms, coords3d_A):
    """
    Return a 3N x 3N Cartesian Hessian (atomic units).
    Replace this stub with your DFT-level predictor.

    coords3d_A: (N,3) in Å
    Return units: Hartree / (Å^2) or Hartree / (Bohr^2) — match what your calculator expects.
    """
    N = len(atoms)
    # Very rough placeholder: small diagonal curvature as a safety net
    H = np.eye(3 * N) * 0.1
    return H


def regularize_minimum_hessian(H, eps=1e-6):
    """Make sure Hessian is positive definite for minima (eigendecomp + floor)."""
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


# --------------------------
#  Calculator wrappers
# --------------------------


class CountingCalc(Calculator):
    """
    Wrap any Calculator; count energy/gradient/Hessian calls.
    """

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.energy_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0

    # Delegate / count
    def get_energy(self, atoms, coords, **kw):
        self.energy_calls += 1
        return self.inner.get_energy(atoms, coords, **kw)

    def get_forces(self, atoms, coords, **kw):
        self.grad_calls += 1
        return self.inner.get_forces(atoms, coords, **kw)

    def get_hessian(self, atoms, coords, **kw):
        self.hess_calls += 1
        return self.inner.get_hessian(atoms, coords, **kw)


# --------------------------
#  Optimizer runners
# --------------------------


def run_baseline_fire(geom, trust=None, max_cycles=200, thresh="gau_loose"):
    # first-order baseline (FIRE) — no Hessian involved
    opt = FIRE(geom, max_cycles=max_cycles, thresh=thresh)
    t0 = time.perf_counter()
    opt.run()
    t1 = time.perf_counter()
    steps = getattr(opt, "cur_cycle", None)
    # Gradient evaluations ~= grad_calls counted on calculator
    calc = geom.calculator
    return {
        "name": "baseline_first_order",
        "converged": bool(getattr(opt, "is_converged", False)),
        "steps": int(steps) if steps is not None else None,
        "grad_evals": getattr(calc, "grad_calls", None),
        "wall_time_s": t1 - t0,
    }


def run_rfo(
    geom,
    *,
    hessian_init,
    hessian_update="bfgs",
    hessian_recalc=None,
    trust_radius=0.3,
    max_cycles=200,
    thresh="gau_loose",
):
    # RFO with flexible Hessian policies. hessian_init ∈ {'unit','calc',...}; hessian_recalc = k or None.
    opt = RFOptimizer(
        # geometry
        #     Geometry to be optimized.
        # line_search
        #     Whether to carry out implicit line searches.
        # gediis
        #     Whether to enable GEDIIS.
        # gdiis
        #     Whether to enable GDIIS.
        # gdiis_thresh
        #     Threshold for rms(forces) to enable GDIIS.
        # gediis_thresh
        #     Threshold for rms(step) to enable GEDIIS.
        # gdiis_test_direction
        #     Whether to the overlap of the RFO step and the GDIIS step.
        # max_micro_cycles
        #     Number of restricted-step microcycles. Disabled by default.
        # adapt_step_func
        #     Whether to switch between shifted Newton and RFO-steps.
        # trust_radius
        #     Initial trust radius in whatever unit the optimization is carried out.
        # trust_update
        #     Whether to update the trust radius throughout the optimization.
        # trust_min
        #     Minimum trust radius.
        # trust_max
        #     Maximum trust radius.
        # max_energy_incr
        #     Maximum allowed energy increased after a faulty step. Optimization is
        #     aborted when the threshold is exceeded.
        # hessian_update
        #     Type of Hessian update. Defaults to BFGS for minimizations and Bofill
        #     for saddle point searches.
        # hessian_init
        #     Type of initial model Hessian.
        # hessian_recalc
        #     Recalculate exact Hessian every n-th cycle instead of updating it.
        # hessian_recalc_adapt
        #     Use a more flexible scheme to determine Hessian recalculation. Undocumented.
        # hessian_xtb
        #     Recalculate the Hessian at the GFN2-XTB level of theory.
        # hessian_recalc_reset
        #     Whether to skip Hessian recalculation after reset. Undocumented.
        # small_eigval_thresh
        #     Threshold for small eigenvalues. Eigenvectors belonging to eigenvalues
        #     below this threshold are discardewd.
        # line_search
        #     Whether to carry out a line search. Not implemented by a subclassing
        #     optimizers.
        # alpha0
        #     Initial alpha for restricted-step (RS) procedure.
        # max_micro_cycles
        #     Maximum number of RS iterations.
        # rfo_overlaps
        #     Enable mode-following in RS procedure.
        geom,
        thresh=thresh,
        trust_radius=trust_radius,
        # np.array, .h5 path, calc, fischer, unit, simple
        hessian_init=hessian_init,
        hessian_update=hessian_update,
        hessian_recalc=hessian_recalc,
        line_search=True,
    )
    t0 = time.perf_counter()
    opt.run()
    t1 = time.perf_counter()
    steps = getattr(opt, "cur_cycle", None)
    calc = geom.calculator
    return {
        "name": f"rfo_{hessian_init}_recalc{hessian_recalc if hessian_recalc else 0}",
        "converged": bool(getattr(opt, "is_converged", False)),
        "steps": int(steps) if steps is not None else None,
        "grad_evals": getattr(calc, "grad_calls", None),
        "wall_time_s": t1 - t0,
    }


# --------------------------
#  Main harness
# --------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "xyz",
        help="input geometry in form of .xyz or folder of .xyz files or lmdb path",
    )
    ap.add_argument(
        "--coord",
        default="redund",
        choices=["cart", "redund", "dlc", "tric"],
        help="coordinate system",
    )
    ap.add_argument("--max_samples", type=int, default=1)
    args = ap.parse_args()

    # case 1: is
    if os.path.isfile(args.xyz):
        input_geometries = [args.xyz]
    # case 2: folder of xyz files
    elif os.path.isdir(args.xyz):
        input_geometries = [
            os.path.join(args.xyz, f)
            for f in os.listdir(args.xyz)
            if f.endswith(".xyz")
        ]
    # case 3: lmdb path
    elif args.xyz.endswith(".lmdb"):
        dataset_path = fix_dataset_path(args.xyz)
        dataset = LmdbDataset(dataset_path)
        # dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    for data in dataset:
        # atoms, coords = load_xyz(xyz)

        print(data)

        atoms = data.atom_types.numpy()
        coords = data.coords.numpy() * ANG2BOHR
        initial_dft_hessian = data.hessian.numpy()

        # Build Geometry; pysisyphus expects Bohr
        geom = Geometry(
            atoms, coords, coord_type=args.coord
        )  # RIC('redund') is recommended for molecules.

        # base_calc = LennardJones()
        base_calc = MLFF(
            charge=0,
            ckpt_path="/ssd/Code/ReactBench/ckpt/hesspred/alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956.ckpt",
            config_path="auto",
            device="cpu",
            hessian_method="predict",
            mem=4000,
            method="equiformer",
            mult=1,
            pal=1,
            # out_dir=yaml_dir / OUT_DIR_DEFAULT,
            # 'out_dir': PosixPath('/ssd/Code/ReactBench/runs/equiformer_alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956_data_predict/rxn9/TSOPT/qm_calcs
        )
        print(f"Initialized calc MLFF.model: {base_calc.model.__class__.__name__}")

        # Wrap it so we can count calls and optionally supply H_pred
        counting_calc = CountingCalc(base_calc)
        geom.set_calculator(counting_calc)

        results = []

        # 1) Baseline: first-order optimization (FIRE)
        results.append(run_baseline_fire(geom.copy()))

        # 2) No Hessian: BFGS with non-Hessian initial guess (unit) — pure quasi-Newton
        #    RFOptimizer accepts hessian_init and BFGS updates.                         :contentReference[oaicite:6]{index=6}
        geom2 = Geometry(atoms, coords, coord_type=args.coord)
        geom2.set_calculator(CountingCalc(base_calc))
        results.append(
            run_rfo(
                geom2,
                hessian_init="unit",
                hessian_update="bfgs",
                hessian_recalc=None,
            )
        )

        # 3) Initial-only: RFO+BFGS with DFT Hessian at step 0
        geom3 = Geometry(atoms, coords, coord_type=args.coord)
        geom3.set_calculator(CountingCalc(base_calc))
        results.append(
            run_rfo(
                geom3,
                hessian_init=initial_dft_hessian,
                hessian_update="bfgs",
                hessian_recalc=None,
            )
        )

        # For 3) and 4) we provide your H_pred through the calculator and ask RFOptimizer to pull it:
        #    hessian_init='calc' gets Hessian from the calculator at step 0;
        #    hessian_recalc=k recomputes it every k steps.                               :contentReference[oaicite:7]{index=7}
        # 3) Initial-only
        geom3 = Geometry(atoms, coords, coord_type=args.coord)
        geom3.set_calculator(CountingCalc(base_calc))
        results.append(
            run_rfo(
                geom3,
                hessian_init="calc",
                hessian_update="bfgs",
                hessian_recalc=None,
            )
        )

        # 4a) Periodic replace: k=3
        geom4 = Geometry(atoms, coords, coord_type=args.coord)
        geom4.set_calculator(CountingCalc(base_calc))
        results.append(
            run_rfo(
                geom4,
                hessian_init="calc",
                hessian_update="bfgs",
                hessian_recalc=3,
            )
        )

        # 4b) Periodic replace: k=1 (every step)
        geom5 = Geometry(atoms, coords, coord_type=args.coord)
        geom5.set_calculator(CountingCalc(base_calc))
        results.append(
            run_rfo(
                geom5,
                hessian_init="calc",
                hessian_update="bfgs",
                hessian_recalc=1,
            )
        )

        # Pretty print
        print("\nStrategy, converged, steps, grad_evals, wall_time_s")
        for r in results:
            print(
                "{:>24s} {:>9s} {:>6} {:>11} {:>12.3f}".format(
                    r["name"],
                    str(r["converged"]),
                    str(r["steps"]),
                    str(r["grad_evals"]),
                    r["wall_time_s"],
                )
            )


if __name__ == "__main__":
    main()
