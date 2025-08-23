import argparse
import time
import sys
import pathlib
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pysisyphus.Geometry import Geometry  # Geometry API + coordinate systems
from pysisyphus.calculators.MLFF import MLFF
from pysisyphus.calculators.Calculator import Calculator  # base class to wrap/override
from pysisyphus.optimizers.FIRE import FIRE  # first-order baseline
from pysisyphus.optimizers.RFOptimizer import RFOptimizer  # second-order RFO + BFGS
from pysisyphus.optimizers.BFGS import BFGS

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path, ROOT_DIR
from torch_geometric.data import DataLoader as TGDataLoader
from nets.prediction_utils import GLOBAL_ATOM_SYMBOLS, GLOBAL_ATOM_NUMBERS, compute_extra_props
import numpy as np
import h5py
import pandas as pd

# from pysisyphus.helpers_pure import eigval_to_wavenumber
# from pysisyphus.helpers import _do_hessian
# from pysisyphus.io.hessian import save_hessian
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

[1]: https://pysisyphus.readthedocs.io/en/latest/min_optimization.html "7. Minimization - pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"
[2]: https://pysisyphus.readthedocs.io/en/latest/coordinate_systems.html "5. Coordinate Systems - pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"
[3]: https://pysisyphus.readthedocs.io/en/latest/calculators.html "6. Calculators - pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"

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
# transform = HessianGraphTransform(
#     cutoff=model.cutoff,
#     max_neighbors=model.max_neighbors,
#     use_pbc=model.use_pbc,
# )
# else:
# transform = None

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
    Return units: Hartree / (Å^2) or Hartree / (Bohr^2) - match what your calculator expects.
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
    
    @property
    def model(self):
        return self.inner.model

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


def run_baseline_fire(geom, trust=None, max_cycles=150, thresh="gau_loose", out_dir="."):
    # first-order baseline (FIRE) - no Hessian involved
    """
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.170201
    Structure optimization algorithm which is 
    significantly faster than standard implementations of the conjugate gradient method 
    and often competitive with more sophisticated quasi-Newton schemes
    It is based on conventional molecular dynamics 
    with additional velocity modifications and adaptive time steps.
    """

    opt = FIRE(
        # Geometry providing coords, forces, energy
        geom, max_cycles=max_cycles, thresh=thresh,
        # Initial time step; adaptively scaled during optimization
        dt=0.1,
        # Maximum allowed time step when increasing dt
        dt_max=1,
        # Consecutive aligned steps before accelerating
        N_acc=2,
        # Factor to increase dt on acceleration
        f_inc=1.1,
        # Factor to reduce mixing a on acceleration; also shrinks dt on reset here
        f_acc=0.99,
        # Unused in this implementation; typical FIRE uses to reduce dt on reset
        f_dec=0.5,
        # Counter of aligned steps since last reset (start at 0)
        n_reset=0,
        # Initial mixing parameter for velocity/force mixing; restored on reset
        a_start=0.1,
        # # Optimizer
        # Convergence threshold.
        # thresh: Thresh = "gau_loose",
        # Maximum absolute component of the allowed step vector. Utilized in
        # optimizers that don't support a trust region or line search.
        # max_step: float = 0.04,
        # Maximum number of allowed optimization cycles.
        # max_cycles: int = 150,
        # Minimum norm of an allowed step. If the step norm drops below
        # this value a ZeroStepLength-exception is raised. The unit depends
        # on the coordinate system of the supplied geometry.
        # min_step_norm: float = 1e-8,
        # Flag that controls whether the norm of the proposed step is check
        # for being too small.
        # assert_min_step: bool = True,
        # rms_force
        # Root-mean-square of the force from which user-defined thresholds
        # are derived. When 'rms_force' is given 'thresh' is ignored.
        # rms_force: Optional[float] = None,
        # When set, convergence is signalled only based on rms(forces).
        # rms_force_only: bool = False,
        # When set, convergence is signalled only based on max(|forces|).
        # max_force_only: bool = False,
        # When set, convergence is signalled only based on max(|forces|) and rms(forces).
        # force_only: bool = False,
        # Threshold for the RMSD with another geometry. When the RMSD drops
        # below this threshold convergence is signalled. Only used with
        # Growing Newton trajectories.
        # converge_to_geom_rms_thresh: float = 0.05,
        # Flag that controls whether the geometry is aligned in every step
        # onto the coordinates of the previous step. Must not be used with
        # internal coordinates.
        # align: bool = False,
        # Factor that controls the strength of the alignment. 1.0 means
        # full alignment, 0.0 means no alignment. The factor mixes the
        # rotation matrix of the alignment with the identity matrix.
        # align_factor: float = 1.0,
        # Flag to control dumping/writing of optimization progress to the
        # filesystem
        # dump: bool = False,
        # Flag to control whether restart information is dumped to the
        # filesystem.
        # dump_restart: bool = False,
        # Report optimization progress every nth cycle.
        # print_every: int = 1,
        # Short string that is prepended to several files created by
        # the optimizer. Allows distinguishing several optimizations carried
        # out in the same directory.
        # prefix: str = "",
        # Controls the minimal allowed similarity between coordinates
        # after two successive reparametrizations. Convergence is signalled
        # if the coordinates did not change significantly.
        # reparam_thresh: float = 1e-3,
        # Whether to check for (too) similar coordinates after reparametrization.
        # reparam_check_rms: bool = True,
        # Reparametrize before or after calculating the step. Can also be turned
        # off by setting it to None.
        # reparam_when: Optional[Literal["before", "after"]] = "after",
        # Signal convergence when max(forces) and rms(forces) fall below the
        # chosen threshold, divided by this factor. Convergence of max(step) and
        # rms(step) is ignored.
        # overachieve_factor: float = 0.0,
        # Check the eigenvalues of the modes we maximize along. Convergence requires
        # them to be negative. Useful if TS searches are started from geometries close
        # to a minimum.
        # check_eigval_structure: bool = False,
        # Restart information. Undocumented.
        # restart_info=None,
        # Whether coordinates of chain-of-sates images are checked for being
        # too similar.
        # check_coord_diffs: bool = True,
        # Unitless threshold for similary checking of COS image coordinates.
        # The first image is assigned 0, the last image is assigned to 1.
        # coord_diff_thresh: float = 0.01,
        # Tuple of lists containing atom indices, defining two fragments.
        # fragments: Optional[Tuple] = None,
        # Monitor fragment distances for N cycles. The optimization is terminated
        # when the interfragment distances falls below the initial value after N
        # cycles.
        # monitor_frag_dists: int = 0,
        # String poiting to a directory where optimization progress is
        # dumped.
        out_dir = out_dir,
        # Basename of the HDF5 file used for dumping.
        # h5_fn: str = "optimization.h5",
        # Groupname used for dumping of this optimization.
        # h5_group_name: str = "opt",
    )
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
    out_dir=".",
):
    # RFO with flexible Hessian policies. hessian_init ∈ {'unit','calc',...}; hessian_recalc = k or None.
    opt = RFOptimizer(
        # geometry
        # Geometry to be optimized.
        # line_search
        # Whether to carry out implicit line searches.
        # gediis
        # Whether to enable GEDIIS.
        # gdiis
        # Whether to enable GDIIS.
        # gdiis_thresh
        # Threshold for rms(forces) to enable GDIIS.
        # gediis_thresh
        # Threshold for rms(step) to enable GEDIIS.
        # gdiis_test_direction
        # Whether to the overlap of the RFO step and the GDIIS step.
        # max_micro_cycles
        # Number of restricted-step microcycles. Disabled by default.
        # adapt_step_func
        # Whether to switch between shifted Newton and RFO-steps.
        # trust_radius
        # Initial trust radius in whatever unit the optimization is carried out.
        # trust_update
        # Whether to update the trust radius throughout the optimization.
        # trust_min
        # Minimum trust radius.
        # trust_max
        # Maximum trust radius.
        # max_energy_incr
        # Maximum allowed energy increased after a faulty step. Optimization is
        # aborted when the threshold is exceeded.
        # hessian_update
        # Type of Hessian update. Defaults to BFGS for minimizations and Bofill
        # for saddle point searches.
        # hessian_init
        # Type of initial model Hessian.
        # hessian_recalc
        # Recalculate exact Hessian every n-th cycle instead of updating it.
        # hessian_recalc_adapt
        # Use a more flexible scheme to determine Hessian recalculation. Undocumented.
        # hessian_xtb
        # Recalculate the Hessian at the GFN2-XTB level of theory.
        # hessian_recalc_reset
        # Whether to skip Hessian recalculation after reset. Undocumented.
        # small_eigval_thresh
        # Threshold for small eigenvalues. Eigenvectors belonging to eigenvalues
        # below this threshold are discardewd.
        # line_search
        # Whether to carry out a line search. Not implemented by a subclassing
        # optimizers.
        # alpha0
        # Initial alpha for restricted-step (RS) procedure.
        # max_micro_cycles
        # Maximum number of RS iterations.
        # rfo_overlaps
        # Enable mode-following in RS procedure.
        geom,
        thresh=thresh,
        trust_radius=trust_radius,
        # np.array, .h5 path, calc, fischer, unit, simple
        hessian_init=hessian_init,
        hessian_update=hessian_update,
        hessian_recalc=hessian_recalc,
        line_search=True,
        out_dir=out_dir,
    )
    t0 = time.perf_counter()
    opt.run()
    t1 = time.perf_counter()
    steps = getattr(opt, "cur_cycle", None)
    calc = geom.calculator
    if isinstance(hessian_init, np.ndarray):
        hessian_init = "passed"
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

def print_header(i, method):
    print("\n" + "="*10 + " " + str(i) + " " + method + " " + "="*10)

def do_relaxations():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xyz",
        default="ts1x-val.lmdb",
        help="input geometry in form of .xyz or folder of .xyz files or lmdb path",
        type=str,
        required=False,
    )
    ap.add_argument(
        "--coord",
        default="redund",
        choices=["cart", "redund", "dlc", "tric"],
        help="coordinate system",
        type=str,
        required=False,
    )
    ap.add_argument("--max_samples", type=int, default=15)
    ap.add_argument("--debug", type=bool, default=False)
    ap.add_argument("--redo", type=bool, default=False)
    args = ap.parse_args()

    ckpt_path = "/ssd/Code/ReactBench/ckpt/hesspred/alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956.ckpt"
    wandb_id = ckpt_path.split("/")[-1].split(".")[0].split("-")[1]

    max_cycles = 150
    if args.debug:
        args.max_samples = 3
        max_cycles = 3

    print("Loading dataset...")
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

    # Determine source label for logging
    source_label = dataset_path.split("/")[-1].split(".")[0]
    out_dir = os.path.join(ROOT_DIR, "runs_relaxation", source_label + "_" + args.coord + "_" + wandb_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Out directory: {out_dir}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(out_dir, f"relaxation_results_{args.max_samples}.csv")

    if not os.path.exists(csv_path) or args.redo:
        # Accumulate results across all samples
        all_results = []

        print()
        for i, data in tqdm(enumerate(dataset), total=args.max_samples):
            if i >= args.max_samples:
                break
            print("", "="*80, i, "="*80, sep="\n")
            # atoms, coords = load_xyz(xyz)

            indices = data.one_hot.long().argmax(dim=1)
            atomssymbols = GLOBAL_ATOM_SYMBOLS[indices.cpu().numpy()]

            coords = data.pos.numpy() * ANG2BOHR
            initial_dft_hessian = data.hessian.numpy()

            # Build Geometry; pysisyphus expects Bohr
            geom = Geometry(
                atomssymbols, coords, coord_type=args.coord
            )  # RIC('redund') is recommended for molecules.

            # base_calc = LennardJones()
            base_calc = MLFF(
                charge=0,
                ckpt_path=ckpt_path,
                config_path="auto",
                device="cuda",
                hessian_method="predict",
                mem=4000,
                method="equiformer",
                mult=1,
                pal=1,
                # out_dir=yaml_dir / OUT_DIR_DEFAULT,
                # 'out_dir': PosixPath('/ssd/Code/ReactBench/runs/equiformer_alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956_data_predict/rxn9/TSOPT/qm_calcs
            )
            print(f"Initialized calc MLFF.model: {base_calc.model.__class__.__name__}")
            print(f"Initialized calc MLFF.model.model: {base_calc.model.model.__class__.__name__}")
            print(f"Initialized calc MLFF.model.model.potential: {base_calc.model.model.potential.__class__.__name__}")

            # Wrap it so we can count calls and optionally supply H_pred
            counting_calc = CountingCalc(base_calc)
            geom.set_calculator(counting_calc)

            results = []

            # 1) Baseline: first-order optimization (FIRE)
            _method = "FIRE"
            print_header(i, _method)
            geom1 = geom.copy_all()
            results.append(
                run_baseline_fire(
                    geom1,
                    max_cycles=max_cycles,
                    out_dir=os.path.join(out_dir, _method),
                )
            )
            results[-1]["human_name"] = _method

            # 2) No Hessian: BFGS with non-Hessian initial guess (unit) - pure quasi-Newton
            #    RFOptimizer accepts hessian_init and BFGS updates.
            _method = "RFO-BFGS (unit init)"
            print_header(i, _method)
            # geom2 = Geometry(atomssymbols, coords, coord_type=args.coord)
            # geom2.set_calculator(CountingCalc(base_calc))
            geom2 = geom.copy_all()
            results.append(
                run_rfo(
                    geom2,
                    hessian_init="unit",
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=os.path.join(out_dir, _method),
                )
            )
            results[-1]["human_name"] = _method

            # 3) Initial-only: RFO+BFGS with DFT Hessian at step 0
            _method = "RFO-BFGS (DFT init)"
            print_header(i, _method)
            # geom3 = Geometry(atomssymbols, coords, coord_type=args.coord)
            # geom3.set_calculator(CountingCalc(base_calc))
            geom3 = geom.copy_all()
            results.append(
                run_rfo(
                    geom3,
                    hessian_init=initial_dft_hessian,
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=os.path.join(out_dir, _method),
                )
            )
            results[-1]["human_name"] = _method

            # we provide your H_pred through the calculator and ask RFOptimizer to pull it:
            #    hessian_init='calc' gets Hessian from the calculator at step 0;
            #    hessian_recalc=k recomputes it every k steps.
            
            # 4) Initial-only: RFO+BFGS with Hpred only at step 0
            _method = "RFO-BFGS (Hpred init)"
            print_header(i, _method)
            # geom3 = Geometry(atomssymbols, coords, coord_type=args.coord)
            # geom3.set_calculator(CountingCalc(base_calc))
            geom3 = geom.copy_all()
            results.append(
                run_rfo(
                    geom3,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=os.path.join(out_dir, _method),
                )
            )
            results[-1]["human_name"] = _method

            # 5) Periodic replace: k=3
            _method = "RFO-BFGS (Hpred k3)"
            print_header(i, _method)
            # geom4 = Geometry(atomssymbols, coords, coord_type=args.coord)
            # geom4.set_calculator(CountingCalc(base_calc))
            geom4 = geom.copy_all()
            results.append(
                run_rfo(
                    geom4,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=3,
                    out_dir=os.path.join(out_dir, _method),
                )
            )
            results[-1]["human_name"] = _method

            # 6) Periodic replace: k=1 (every step)
            _method = "RFO (Hpred)"
            print_header(i, _method)
            # geom5 = Geometry(atomssymbols, coords, coord_type=args.coord)
            # geom5.set_calculator(CountingCalc(base_calc))
            geom5 = geom.copy_all()
            results.append(
                run_rfo(
                    geom5,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=os.path.join(out_dir, _method),
                )
            )
            results[-1]["human_name"] = _method

            # Pretty print
            print("\nStrategy, converged, steps, grad_evals, wall_time_s")
            for r in results:
                print(
                    f"{r['human_name']:>24s} {r['name']:>15s} {args.coord:>6} {str(r['converged']):>6} {str(r['steps']):>11} {str(r['grad_evals']):>12s} {r['wall_time_s']:>12.3f}"
                )

            # Collect results with context for CSV
            for r in results:
                r_with_ctx = dict(r)
                r_with_ctx.update({
                    "sample_index": i,
                    "coord": args.coord,
                    "source": source_label,
                })
                all_results.append(r_with_ctx)

        # Write aggregated CSV
        if len(all_results) > 0:
            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)
            print(f"\nSaved relaxation results to: {csv_path}")
        else:
            print(f"\nNo results to save to {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"\nLoaded relaxation results from: {csv_path}")

    # print mean and std for each method and metric
    print("\nMean and std for each method and metric:")
    for metric in ["steps", "grad_evals", "wall_time_s"]:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for method in df["human_name"].unique():
            _d = df[df["human_name"] == method]
            print(f"{method}: {_d[metric].mean():.2f} ± {_d[metric].std():.2f}")

    #########################################################
    # do plotting
    #########################################################
    sns.set_theme(style="whitegrid")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _plot_metric_box(_df, metric_name, save_path):
        _d = _df.dropna(subset=[metric_name])
        if len(_d) == 0:
            return
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=_d,
            x="human_name",
            y=metric_name,
            # palette="tab10",
            # hue=
            # width=0.5,
        )
        ax.set_xlabel("Method")
        ax.set_ylabel(metric_name.replace("_", " "))
        ax.set_title(f"{metric_name.replace('_', ' ').title()} by method")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved {save_path}")

    for metric in ["steps", "grad_evals", "wall_time_s"]:
        _plot_metric_box(df, metric, os.path.join(plots_dir, f"{metric}.png"))

    # Convergence rate per method
    if "converged" in df.columns:
        conv = (
            df.groupby("human_name")["converged"]
            .mean()
            .reset_index(name="convergence_rate")
        )
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=conv,
            x="human_name",
            y="convergence_rate",
            # palette="tab10",
            # hue="coord",
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Method")
        ax.set_ylabel("Convergence rate")
        ax.set_title("Convergence rate by method")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fname = os.path.join(plots_dir, "convergence_rate.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")

    print(f"Saved plots to: {plots_dir}")

    return df

if __name__ == "__main__":
    df = do_relaxations()

