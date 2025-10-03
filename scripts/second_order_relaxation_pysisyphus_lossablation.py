import argparse
import time
import sys
import pathlib
import contextlib
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import logging
import h5py
import pandas as pd
import wandb
import shutil

import torch

try:
    from pysisyphus.Geometry import Geometry  # Geometry API + coordinate systems

    # from pysisyphus.calculators.MLFF import MLFF
    from pysisyphus.calculators.Calculator import (
        Calculator,
    )  # base class to wrap/override

    # from pysisyphus.optimizers.FIRE import FIRE  # first-order baseline
    # from pysisyphus.optimizers.BFGS import BFGS
    # from pysisyphus.optimizers.SteepestDescent import SteepestDescent
    # from pysisyphus.optimizers.ConjugateGradient import ConjugateGradient
    from pysisyphus.optimizers.RFOptimizer import RFOptimizer  # second-order RFO + BFGS
    from pysisyphus.optimizers.BacktrackingOptimizer import BacktrackingOptimizer

    # from pysisyphus.helpers_pure import eigval_to_wavenumber
    # from pysisyphus.helpers import _do_hessian
    # from pysisyphus.io.hessian import save_hessian
    from pysisyphus.constants import AU2EV, BOHR2ANG
    from pysisyphus.helpers import procrustes

    from ReactBench.Calculators.equiformer import PysisEquiformer
except ImportError:
    print()
    traceback.print_exc()
    print("\nFollow the instructions here: https://github.com/BurgerAndreas/ReactBench")
    exit()

from gadff.path_config import fix_dataset_path, ROOT_DIR
# from gadff.horm.ff_lmdb import LmdbDataset
# from nets.prediction_utils import (
#     GLOBAL_ATOM_SYMBOLS,
#     GLOBAL_ATOM_NUMBERS,
#     compute_extra_props,
# )

from gadff.t1x_dft_dataloader import Dataloader as T1xDFTDataloader

# try:
#     from transition1x import Dataloader as T1xDataloader
# except ImportError:
#     print(
#         "Transition1x not found, please install it by:\n"
#         "git clone https://gitlab.com/matschreiner/Transition1x.git" + "\n"
#         "uv run Transition1x/download_t1x.py Transition1x/data" + "\n"
#         "uv pip install -e Transition1x" + "\n"
#     )
from gadff.colours import (
    COLOUR_LIST,
    OPTIM_TO_COLOUR,
    ANNOTATION_FONT_SIZE,
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRIC_TO_LABEL = {
    "steps": "Steps to Convergence",
    "wall_time_s": "Wall Time [s]",
}


pysis_all_optimizers = [
    "BFGS",
    "ConjugateGradient",
    "CubicNewton",
    "FIRE",
    "LayerOpt",
    "LBFGS",
    "MicroOptimizer",
    "NCOptimizer",
    "PreconLBFGS",
    "PreconSteepestDescent",
    "QuickMin",
    "RFOptimizer",
    "SteepestDescent",
    "StringOptimizer",
    "StabilizedQNMethod",
]

# --------------------------
#  Utilities
# --------------------------


Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


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


def clean_str(s):
    return "".join(c for c in s.replace(" ", "_") if c.isalnum()).lower()


COORD_TO_NAME = {
    "cart": "Cartesian Coordinates",
    "redund": "Redundant Internal Coordinates",
    "tric": "Translation & Rotation Internal Coordinates",
    "dlc": "Delocalized Internal Coordinates",
}


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

    def __init__(self, inner, assert_pd_hessians=False, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.reset()
        self.assert_pd_hessians = assert_pd_hessians

    def reset(self):
        self.energy_calls = 0
        self.grad_calls = 0
        self.hessian_calls = 0
        self.calculate_calls = 0
        self.calculate_energy_calls = 0
        self.calculate_gradient_calls = 0
        self.calculate_hessian_calls = 0
        self.cnt_not_pd = 0
        super().reset()

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
        self.hessian_calls += 1
        results = self.inner.get_hessian(atoms, coords, **kw)
        # check if hessian is positive definite
        if not np.all(np.linalg.eigvals(results["hessian"]) > 0):
            print("Predicted Hessian is not positive definite")
            self.cnt_not_pd += 1
            if self.assert_pd_hessians:
                raise ValueError("Predicted Hessian is not positive definite")
        return results

    def get_num_hessian(self, atoms, coords, prepare_kwargs={}):
        self.hessian_calls += 1
        results = self.inner.get_num_hessian(atoms, coords, **prepare_kwargs)
        # check if hessian is positive definite
        if not np.all(np.linalg.eigvals(results["hessian"]) > 0):
            print("Numerical Hessian is not positive definite")
            self.cnt_not_pd += 1
            if self.assert_pd_hessians:
                raise ValueError("Numerical Hessian is not positive definite")
        return results

    def calculate(self, atom=None, properties=None, **kwargs):
        self.calculate_calls += 1
        if properties is None:
            properties = kwargs.get("properties", None)
        if properties is not None:
            if "energy" in properties:
                self.calculate_energy_calls += 1
            if "gradient" in properties:
                self.calculate_gradient_calls += 1
            if "hessian" in properties:
                self.calculate_hessian_calls += 1
        return self.inner.calculate(atom, properties, **kwargs)


# --------------------------
#  Optimizer runners
# --------------------------


class NaiveSteepestDescent(BacktrackingOptimizer):
    def __init__(self, geometry, **kwargs):
        super(NaiveSteepestDescent, self).__init__(geometry, alpha=0.1, **kwargs)

    def optimize(self):
        if self.is_cos and self.align:
            procrustes(self.geometry)

        self.forces.append(self.geometry.forces)

        step = self.alpha * self.forces[-1]
        step = self.scale_by_max_step(step)
        return step


def _run_opt_safely(
    geom,
    opt,
    method_name,
    out_dir,
    verbose=False,
    start_clean=True,
    dft_hessian_is_pd=True,
):
    # logging
    if start_clean:
        geom.calculator.reset()
        assert geom.calculator.grad_calls == 0, (
            f"Calculator counts {geom.calculator.grad_calls} gradient calls"
        )
        # assert geom._masses is None, f"Masses are not None: {geom._masses}" # computed in Geometry.__init__
        assert geom._energy is None, f"Energy is not None: {geom._energy}"
        assert geom._forces is None, f"Forces are not None: {geom._forces}"
        assert geom._hessian is None, f"Hessian is not None: {geom._hessian}"
        assert geom._all_energies is None, (
            f"All energies are not None: {geom._all_energies}"
        )

    method_name_clean = clean_str(method_name)
    log_path = os.path.join(out_dir, f"optrun_{method_name_clean}.txt")

    # wrapper to run optimizer and return results
    def _try_to_run(_opt):
        try:
            t0 = time.perf_counter()
            _opt.run()
            t1 = time.perf_counter()
            steps = _opt.cur_cycle
            return {
                "name": method_name,
                "converged": bool(getattr(_opt, "is_converged", False)),
                "steps": int(steps) if steps is not None else None,
                "grad_calls": geom.calculator.grad_calls,
                "hessian_calls": geom.calculator.hessian_calls,
                "energy_calls": geom.calculator.energy_calls,
                "calculate_calls": geom.calculator.calculate_calls,
                "calculate_energy_calls": geom.calculator.calculate_energy_calls,
                "calculate_gradient_calls": geom.calculator.calculate_gradient_calls,
                "calculate_hessian_calls": geom.calculator.calculate_hessian_calls,
                "cnt_not_pd": geom.calculator.cnt_not_pd,
                "wall_time_s": t1 - t0,
                "dft_hessian_is_pd": dft_hessian_is_pd,
            }
        except Exception as e:
            print(f"Error running {method_name} optimization: {e}", flush=True)
            traceback.print_exc()
            return {
                "name": method_name,
                "converged": False,
                "steps": None,
                "grad_calls": None,
                "hessian_calls": None,
                "energy_calls": None,
                "calculate_calls": None,
                "calculate_energy_calls": None,
                "calculate_gradient_calls": None,
                "calculate_hessian_calls": None,
                "cnt_not_pd": None,
                "wall_time_s": None,
                "dft_hessian_is_pd": dft_hessian_is_pd,
            }

    # run optimizer and return results
    if verbose:
        return _try_to_run(opt)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"Saving log to {log_path}")
        with open(log_path, "a") as _log_fh, contextlib.redirect_stdout(
            _log_fh
        ), contextlib.redirect_stderr(_log_fh):
            return _try_to_run(opt)


def get_rfo_optimizer(
    geom,
    *,
    hessian_init,
    thresh,
    hessian_update="bfgs",
    hessian_recalc=None,
    trust_radius=0.3,
    max_cycles=200,
    out_dir=".",
    verbose=False,
):
    # RFO with flexible Hessian policies. hessian_init ∈ {'unit','calc',...}; hessian_recalc = k or None.
    opt = RFOptimizer(
        # line_search: bool = True
        # Whether to carry out implicit line searches.
        # gediis: bool = False
        # Whether to enable GEDIIS.
        # gdiis: bool = True
        # Whether to enable GDIIS.
        # gdiis_thresh: float = 2.5e-3
        # Threshold for rms(forces) to enable GDIIS.
        # gediis_thresh: float = 1e-2
        # Threshold for rms(step) to enable GEDIIS.
        # gdiis_test_direction: bool = True
        # Whether to the overlap of the RFO step and the GDIIS step.
        # max_micro_cycles: int = 25
        # Number of restricted-step microcycles. Disabled by default.
        # adapt_step_func: bool = False
        # # HessianOptimizer
        # Whether to switch between shifted Newton and RFO-steps.
        # trust_radius: float = 0.5
        # Initial trust radius in whatever unit the optimization is carried out.
        # trust_update: bool = True
        # Whether to update the trust radius throughout the optimization.
        # trust_min: float = 0.1
        # Minimum trust radius.
        # trust_max: float = 1
        # Maximum trust radius.
        # max_energy_incr: Optional[float] = None
        # Maximum allowed energy increased after a faulty step. Optimization is
        # aborted when the threshold is exceeded.
        # hessian_update: HessUpdate = "bfgs"
        # Type of Hessian update. Defaults to BFGS for minimizations and Bofill
        # for saddle point searches.
        # hessian_init: HessInit = "fischer"
        # Type of initial model Hessian.
        # hessian_recalc: Optional[int] = None
        # Recalculate exact Hessian every n-th cycle instead of updating it.
        # hessian_recalc_adapt: Optional[float] = None
        # Use a more flexible scheme to determine Hessian recalculation. Undocumented.
        # hessian_xtb: bool = False
        # Recalculate the Hessian at the GFN2-XTB level of theory.
        # hessian_recalc_reset: bool = False
        # Whether to skip Hessian recalculation after reset. Undocumented.
        # small_eigval_thresh: float = 1e-8
        # Threshold for small eigenvalues. Eigenvectors belonging to eigenvalues
        # below this threshold are discardewd.
        # line_search: bool = False
        # Whether to carry out a line search. Not implemented by a subclassing
        # optimizers.
        # alpha0: float = 1.0
        # Initial alpha for restricted-step (RS) procedure.
        # max_micro_cycles: int = 25
        # Maximum number of RS iterations.
        # rfo_overlaps: bool = False
        # Enable mode-following in RS procedure.
        # Geometry to be optimized.
        geom,
        thresh=thresh,
        trust_radius=trust_radius,
        # np.array, .h5 path, calc, fischer, unit, simple
        hessian_init=hessian_init,
        hessian_update=hessian_update,
        hessian_recalc=hessian_recalc,
        line_search=True,
        out_dir=out_dir,
        max_cycles=max_cycles,
        # # TS opt in ReactBench uses
        # # pysisyphus.tsoptimizers.RSPRFOptimizer
        # type: rsprfo
        # do_hess: True
        # thresh: gau
        # max_cycles: 50
        # trust_radius: 0.2 # here we use 0.3
        # hessian_recalc: 1
        allow_write=False,
    )
    return opt


# --------------------------
#  Main harness
# --------------------------


def get_geom(atomssymbols, coords, coord_type, base_calc_mae, args):
    geom = Geometry(atomssymbols, coords, coord_type=coord_type)
    base_calc_mae.reset()
    counting_calc = CountingCalc(base_calc_mae, assert_pd_hessians=False)
    geom.set_calculator(counting_calc)
    return geom


def print_header(i, method):
    print("\n" + "=" * 10 + " " + str(i) + " " + method + " " + "=" * 10)


# match OPTIM_TO_COLOUR
METHOD_TO_CATEGORY = {
    "NaiveSteepestDescent": "First-Order",
    "SteepestDescent": "First-Order",
    "FIRE": "First-Order",
    "ConjugateGradient": "First-Order",
    "RFO-BFGS (unit init)": "Quasi-Second-Order",
    "RFO-BFGS (DFT init)": "Quasi-Second-Order",
    "RFO-BFGS (autograd init)": "Quasi-Second-Order",
    "RFO-BFGS (NumHess init)": "Quasi-Second-Order",
    "RFO-BFGS (learned init)": "Quasi-Second-Order",
    "RFO-BFGS (learned k3)": "Quasi-Second-Order",
    "RFO (NumHess)": "Second-Order",
    "RFO (NumHess 4)": "Second-Order",
    "RFO (autograd)": "Second-Order",
    # "RFO (learned)": "ours",
    "RFO (learned)": "Second-Order",
}
rename_categories = {
    "First-Order": "No Hessians",
    "Quasi-Second-Order": "Quasi-Hessian",
    "Second-Order": "Hessian",
}
METHOD_TO_CATEGORY = {k: rename_categories[v] for k, v in METHOD_TO_CATEGORY.items()}
METHOD_TO_COLOUR = {
    m: OPTIM_TO_COLOUR[METHOD_TO_CATEGORY[m]] for m in METHOD_TO_CATEGORY
}
DO_METHOD = [
    "RFO (predicted MSE)",
    "RFO (predicted MAE)",
    "RFO (predicted MAE+Sub)",
]

# Plot again
COMPETATIVE_METHODS_STEPS = [
    "RFO (predicted MSE)",
    "RFO (predicted MAE)",
    "RFO (predicted MAE+Sub)",
]
COMPETATIVE_METHODS_WALL_TIME = [
    "RFO (predicted MSE)",
    "RFO (predicted MAE)",
    "RFO (predicted MAE+Sub)",
]

RENAME_METHODS_PLOT = {
    "NaiveSteepestDescent": "SteepestDescent",
    "RFO-BFGS (NumHess init)": "RFO-BFGS (FiniteDifference init)",
    "RFO (NumHess)": "RFO (FiniteDifference)",
}


def do_relaxations(out_dir, source_label, args):
    print("Loading dataset...")
    print(f"Dataset: {args.xyz}. is file: {os.path.isfile(args.xyz)}")

    # dataset_path = "Transition1x/data/transition1x.h5"
    noise_str = "0.05"
    dataset_path = (
        f"../Datastore/t1x/t1x_val_reactant_hessian_100_noiserms{noise_str}.h5"
    )
    dataset = T1xDFTDataloader(dataset_path, datasplit="val", only_final=True)
    dataset_path = (
        f"data/t1x_val_reactant_hessian_100_noiserms{noise_str.replace('.', '')}.h5"
    )

    try:
        len_dataset = len(dataset)
    except:
        len_dataset = 1_000

    if args.max_samples > len_dataset:
        args.max_samples = len_dataset

    if args.redo:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Out directory: {out_dir}")

    rng = np.random.default_rng(seed=42)

    print("\nRunning relaxations...")
    csv_path = os.path.join(out_dir, f"relaxation_results.csv")
    if not os.path.exists(csv_path) or args.redo:
        print("\nInitializing model...")
        base_calc_mae = PysisEquiformer(
            charge=0,
            ckpt_path="ckpt/eq_l1_mae.ckpt",
            config_path="auto",
            device="cuda",
            hessianmethod_name="predict",
            hessian_method="predict",  # "autograd", "predict"
            mem=4000,
            method="equiformer",
            mult=1,
            pal=1,
            # out_dir=yaml_dir / OUT_DIR_DEFAULT,
            # 'out_dir': PosixPath('/ssd/Code/ReactBench/runs/equiformer_alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956_data_predict/rxn9/TSOPT/qm_calcs
        )
        base_calc_mse = PysisEquiformer(
            charge=0,
            ckpt_path="ckpt/eq_l1_mse.ckpt",
            config_path="auto",
            device="cuda",
            hessianmethod_name="predict",
            hessian_method="predict",  # "autograd", "predict"
            mem=4000,
            method="equiformer",
            mult=1,
            pal=1,
            # out_dir=yaml_dir / OUT_DIR_DEFAULT,
            # 'out_dir': PosixPath('/ssd/Code/ReactBench/runs/equiformer_alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956_data_predict/rxn9/TSOPT/qm_calcs
        )
        base_calc_maesub = PysisEquiformer(
            charge=0,
            ckpt_path="ckpt/eq_l1_luca8mae.ckpt",
            config_path="auto",
            device="cuda",
            hessianmethod_name="predict",
            hessian_method="predict",  # "autograd", "predict"
            mem=4000,
            method="equiformer",
            mult=1,
            pal=1,
            # out_dir=yaml_dir / OUT_DIR_DEFAULT,
            # 'out_dir': PosixPath('/ssd/Code/ReactBench/runs/equiformer_alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956_data_predict/rxn9/TSOPT/qm_calcs
        )

        print("\nTesting model with pysisyphus...")
        counting_calc = CountingCalc(base_calc_mae)
        molecule = next(iter(dataset))
        if "positions_noised" in molecule["reactant"]:
            coords = molecule["reactant"]["positions_noised"]
        else:
            coords = molecule["reactant"]["positions"]
        # ts = molecule["transition_state"]["positions"]
        # product = molecule["product"]["positions"]
        atoms = np.array(molecule["reactant"]["atomic_numbers"])
        atomssymbols = [Z_TO_SYMBOL[a] for a in atoms]
        coords = coords / BOHR2ANG  # same as *ANG2BOHR
        t1xdataloader = iter(dataset)
        geom = Geometry(atomssymbols, coords, coord_type=args.coord)
        geom.set_calculator(counting_calc)
        energy = geom.energy
        forces = geom.forces
        hessian = geom.hessian
        # Test finished

        ts = time.strftime("%Y%m%d-%H%M%S")
        # Accumulate results across all samples
        all_results = []

        np.random.seed(42)
        torch.manual_seed(42)
        random_idx = np.random.permutation(len_dataset)

        print()
        optims_done = 0
        for cnt, idx in enumerate(random_idx):
            if optims_done >= args.max_samples:
                break
            print(
                "",
                "=" * 80,
                f"\tSample {optims_done} (tried cnt={cnt}, idx={idx} / {len_dataset})\t",
                "=" * 80,
                sep="\n",
            )
            molecule = next(t1xdataloader)
            idx = molecule["reactant"].get("idx", cnt)
            if "positions_noised" in molecule["reactant"]:
                coords = molecule["reactant"]["positions_noised"]
                print("Using noised geometry")
            else:
                coords = molecule["reactant"]["positions"]
                print("Using original geometry")
            atoms = np.array(molecule["reactant"]["atomic_numbers"])
            atomssymbols = [Z_TO_SYMBOL[a] for a in atoms]
            coords = coords / BOHR2ANG  # same as *ANG2BOHR
            initial_dft_hessian = molecule["reactant"]["wB97x_6-31G(d).hessian"]
            # eV/Angstrom^2 -> Hartree/Bohr^2
            # initial_dft_hessian = initial_dft_hessian * AU2EV * BOHR2ANG * BOHR2ANG

            results = []

            # we provide your H_pred through the calculator and ask RFOptimizer to pull it:
            #    hessian_init='calc' gets Hessian from the calculator at step 0;
            #    hessian_recalc=k recomputes it every k steps.

            # Periodic replace: k=1 (every step)
            method_name = "RFO (predicted MSE)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfolearned = get_geom(
                    atomssymbols, coords, args.coord, base_calc_mse, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfolearned,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=False,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_rfolearned,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=False,
                    )
                )
            method_name = "RFO (predicted MAE)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfolearned = get_geom(
                    atomssymbols, coords, args.coord, base_calc_mae, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfolearned,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=False,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_rfolearned,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=False,
                    )
                )
            method_name = "RFO (predicted MAE+Sub)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfolearned = get_geom(
                    atomssymbols, coords, args.coord, base_calc_maesub, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfolearned,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=False,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_rfolearned,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=False,
                    )
                )

            ###########################
            # Pretty print
            print(
                f"\n{'Strategy':>24s} {'coords':>6} {'converged':>6} {'steps':>6} {'grads':>6} {'hessians':>6} {'s':>6}"
            )
            for r in results:
                try:
                    print(
                        f"{r['name']:>24s} {args.coord:>6} {str(r['converged']):>6s} {str(r['steps']):>6s} {str(r['grad_calls']):>6s} {str(r['hessian_calls']):>6s} {r['wall_time_s']:>12.3f}"
                    )
                except:
                    print(f"Error printing {r}")
                    print(r)

            # print positive definite status extra
            print(f"cnt_not_pd:")
            for r in results:
                _msg = f"{r['name']}: {r['cnt_not_pd']}"
                if r["hessian_calls"] is not None and r["hessian_calls"] > 0:
                    _msg += f" ({r['cnt_not_pd'] / r['hessian_calls'] * 100:.2f}%)"
                print(_msg)

            # Collect results with context for CSV
            for r in results:
                r_with_ctx = dict(r)
                r_with_ctx.update(
                    {
                        "sample_index": idx,
                        "coord": args.coord,
                        "source": source_label,
                    }
                )
                all_results.append(r_with_ctx)

            optims_done += 1

        #########################################################
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

    return df


def plot_results(df, out_dir, args):
    # remove all rows where the method is not in DO_METHOD
    df = df[df["name"].isin(DO_METHOD)]

    # print mean and std for each method and metric
    print("\nMean and std for each method and metric:")
    for metric in [
        "steps",
        "grad_calls",
        "hessian_calls",
        # "energy_calls",
        # "calculate_calls",
        # "calculate_energy_calls",
        # "calculate_gradient_calls",
        # "calculate_hessian_calls",
        "cnt_not_pd",
        "wall_time_s",
    ]:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for method in df["name"].unique():
            _d = df[df["name"] == method]
            print(f"{method}: {_d[metric].mean():.2f} ± {_d[metric].std():.2f}")

    #########################################################
    # do plotting
    #########################################################
    print()
    sns.set_theme(style="whitegrid")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _remove_outliers(_d, metric_name):
        # remove the most extreme outliers (1st and 99th percentiles)
        low, high = _d[metric_name].quantile([0.01, 0.99])
        len_before = len(_d)
        _d = _d[(_d[metric_name] >= low) & (_d[metric_name] <= high)]
        len_after = len(_d)
        print(f"Removed {len_before - len_after} outliers for {metric_name}")
        return _d

    def _hex_to_rgba(hex_color, alpha):
        try:
            hc = str(hex_color).lstrip("#")
            r = int(hc[0:2], 16)
            g = int(hc[2:4], 16)
            b = int(hc[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return hex_color

    def _prepare_order(dfin, metric):
        dfo = dfin.copy()
        # if (
        #     metric in ("steps", "wall_time_s")
        #     and "RFO (predicted MAE+Sub)" in dfo["name"].unique()
        # ):
        #     mask = dfo["name"] == "RFO (learned)"
        #     k = int(min(5, mask.sum()))
        #     if k > 0:
        #         idxs = dfo.loc[mask, metric].nlargest(k).index
        #         dfo = dfo.drop(idxs)
        return (
            dfo.groupby("name")[metric]
            # .mean()
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def _series_for_method(dfin, method, metric):
        s = dfin[dfin["name"] == method][metric].dropna()
        if metric in ("steps", "wall_time_s"):
            k = min(5, len(s))
            if k > 0:
                s = s.drop(s.nlargest(k).index)
        return s

    def _plot_metric_violin_plotly(_df, metric_name, save_path):
        _d = _df.dropna(subset=[metric_name])
        if len(_d) == 0:
            return
        # compute order by descending mean (most left -> least right)
        # after removing outliers for RFO (predicted) a.k.a. "RFO (learned)"
        d_for_order = _d.copy()
        if (
            metric_name in ("steps", "wall_time_s")
            and "RFO (learned)" in d_for_order["name"].unique()
        ):
            mask = d_for_order["name"] == "RFO (learned)"
            k = int(min(5, mask.sum()))
            if k > 0:
                idxs = d_for_order.loc[mask, metric_name].nlargest(k).index
                d_for_order = d_for_order.drop(idxs)
        order = (
            d_for_order.groupby("name")[metric_name]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig = go.Figure()

        display_order = []
        methods_plotted = []
        method_to_display_name = {}
        for method in order:
            series = _d[_d["name"] == method][metric_name].dropna()
            # For RFO (predicted) i.e. method == "RFO (learned)", drop top-5 highest values before plotting
            if metric_name in ("steps", "wall_time_s") and method == "RFO (learned)":
                k = min(5, len(series))
                if k > 0:
                    series = series.drop(series.nlargest(k).index)
            if len(series) == 0:
                continue
            display_name = RENAME_METHODS_PLOT.get(method, method)
            # Rename for Plotly display: learned -> predicted
            if method == "RFO (learned)":
                display_name = "RFO (predicted)"
            elif method == "RFO-BFGS (learned init)":
                display_name = "RFO-BFGS (predicted init)"
            # Keep the (ours) suffix for our methods
            # if method in ("RFO (learned)", "RFO-BFGS (learned init)"):
            #     display_name = f"{display_name} (ours)"
            color = METHOD_TO_COLOUR.get(method, "#1f77b4")
            display_order.append(display_name)
            methods_plotted.append(method)
            method_to_display_name[method] = display_name
            fig.add_trace(
                go.Violin(
                    y=series.astype(float),
                    name=display_name,
                    line_color=color,
                    # Defaults to a half-transparent variant of the line color
                    # fillcolor=color,
                    fillcolor=_hex_to_rgba(color, 0.25),
                    opacity=1.0,
                    box_visible=True,
                    meanline_visible=False,
                    spanmode="hard",
                    points="all",
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(color=color, opacity=0.5, size=4),
                    showlegend=False,
                )
            )
        # Add legend for categories (three colours) using dummy scatter traces
        categories_in_plot = []
        for m in methods_plotted:
            cat = METHOD_TO_CATEGORY.get(m)
            if cat is not None and cat not in categories_in_plot:
                categories_in_plot.append(cat)
        for cat in categories_in_plot:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=OPTIM_TO_COLOUR.get(cat, "#1f77b4"), size=10),
                    name=cat,
                    showlegend=True,
                )
            )

        # # Annotate "ours" over highest values of selected methods
        # target_methods = [
        #     "RFO-BFGS (learned init)",
        #     "RFO (learned)",
        # ]
        # if metric_name in _d.columns and len(_d[metric_name].dropna()) > 0:
        #     y_min = float(_d[metric_name].min())
        #     y_max = float(_d[metric_name].max())
        # else:
        #     y_min = 0.0
        #     y_max = 0.0
        # y_pad = 0.02 * (y_max - y_min) if y_max > y_min else 0.0
        # for method in target_methods:
        #     if method in order:
        #         series_ann = _d[_d["name"] == method][metric_name].dropna()
        #         if len(series_ann) == 0:
        #             continue
        #         y_top = float(series_ann.max())
        #         display_name = method_to_display_name.get(method, method)
        #         fig.add_annotation(
        #             x=display_name,
        #             y=y_top + y_pad,
        #             text="<b>ours</b>",
        #             showarrow=False,
        #             xref="x",
        #             yref="y",
        #             xanchor="center",
        #             yanchor="bottom",
        #             font=dict(size=10),
        #         )

        # Bold our two methods in tick labels
        bold_targets = set()
        # for m in ("RFO (learned)", "RFO-BFGS (learned init)"):
        #     if m in method_to_display_name:
        #         bold_targets.add(method_to_display_name[m])
        ticktext = [
            f"<b>{name}</b>" if name in bold_targets else name for name in display_order
        ]
        fig.update_layout(
            template="plotly_white",
            yaxis_title=METRIC_TO_LABEL.get(
                metric_name.lower(), metric_name.replace("_", " ").title()
            ),
            xaxis_title="",
            xaxis=dict(
                categoryorder="array",
                categoryarray=display_order,
                tickvals=display_order,
                ticktext=ticktext,
                tickangle=-25,
            ),
            legend=dict(
                x=1.0,
                y=1.0,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            showlegend=True,
            height=600,
            width=1000,
            # margin=dict(t=0, b=0, l=0, r=0),
            margin=dict(t=0, b=40, l=20, r=0),
        )
        fig.write_image(save_path, scale=2)
        print(f"Saved\n {save_path}")

    def _plot_metric_violin_plotly_double(_df, save_path):
        """
        Plotly violin plot with two subplots: steps to convergence and wall time
        """
        # Data variants
        df_steps = _df.dropna(subset=["steps"]).copy()
        df_wall_comp = _df.dropna(subset=["wall_time_s"]).copy()
        df_wall_comp = df_wall_comp[
            df_wall_comp["name"].isin(COMPETATIVE_METHODS_WALL_TIME)
        ].copy()

        order_steps = _prepare_order(df_steps, "steps")
        order_wall_comp = _prepare_order(df_wall_comp, "wall_time_s")

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Steps to Convergence",
                "Wall Time [s] (Subset)",
            ),
            horizontal_spacing=0.07,
            vertical_spacing=0.0,
            # 9 vs 6 methods plotted
            column_widths=[1.0, 0.8],
        )

        categories_all = []

        for col_idx, (df_i, metric_i, order_i) in enumerate(
            (
                (df_steps, "steps", order_steps),
                (df_wall_comp, "wall_time_s", order_wall_comp),
            ),
            start=1,
        ):
            if len(df_i) == 0 or len(order_i) == 0:
                continue
            display_order = []
            methods_plotted = []
            method_to_display_name = {}
            for method in order_i:
                series = _series_for_method(df_i, method, metric_i)
                if len(series) == 0:
                    continue
                display_name = RENAME_METHODS_PLOT.get(method, method)
                if method == "RFO (learned)":
                    display_name = "RFO (predicted)"
                elif method == "RFO-BFGS (learned init)":
                    display_name = "RFO-BFGS (predicted init)"
                # if method in ("RFO (learned)", "RFO-BFGS (learned init)"):
                #     display_name = f"{display_name} (ours)"
                color = METHOD_TO_COLOUR.get(method, "#1f77b4")

                display_order.append(display_name)
                methods_plotted.append(method)
                method_to_display_name[method] = display_name

                # Violin plot
                fig.add_trace(
                    go.Violin(
                        y=series.astype(float),
                        name=display_name,
                        line_color=color,
                        fillcolor=_hex_to_rgba(color, 0.1),
                        opacity=1.0,
                        width=0.9,  # fixed width
                        box_visible=False,  # show the boxplot
                        meanline_visible=False,
                        spanmode="hard",
                        points="all",
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(color=color, opacity=0.3, size=4),
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )

                # Overlay median as a horizontal tick marker (no box)
                median_value = float(np.median(series.astype(float)))
                fig.add_trace(
                    go.Scatter(
                        x=[display_name],
                        y=[median_value],
                        mode="markers",
                        marker=dict(
                            symbol="line-ew",  # horizontal line marker
                            size=18,
                            color=color,
                            line=dict(color=color, width=2),
                            opacity=1.0,
                        ),
                        hovertemplate="median: %{y:.3g}<extra></extra>",
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )

            for m in methods_plotted:
                cat = METHOD_TO_CATEGORY.get(m)
                if cat is not None and cat not in categories_all:
                    categories_all.append(cat)

            bold_targets = set()
            for m in ("RFO (learned)", "RFO-BFGS (learned init)"):
                if m in method_to_display_name:
                    bold_targets.add(method_to_display_name[m])
            ticktext = [
                f"<b>{name}</b>" if name in bold_targets else name
                for name in display_order
            ]
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=display_order,
                tickvals=display_order,
                ticktext=ticktext,
                tickangle=-25,
                row=1,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=METRIC_TO_LABEL.get(
                    metric_i.lower(), metric_i.replace("_", " ").title()
                ),
                row=1,
                col=col_idx,
            )

            # Annotate "ours" over highest values of selected methods for this subplot
            target_methods = [
                "RFO-BFGS (learned init)",
                "RFO (learned)",
            ]
            if metric_i in df_i.columns and len(df_i[metric_i].dropna()) > 0:
                y_min_i = float(df_i[metric_i].min())
                y_max_i = float(df_i[metric_i].max())
            else:
                y_min_i = 0.0
                y_max_i = 0.0
            y_pad_i = 0.01 * (y_max_i - y_min_i) if y_max_i > y_min_i else 0.0
            for method in target_methods:
                if method in order_i:
                    series_ann_i = _series_for_method(df_i, method, metric_i)
                    if len(series_ann_i) == 0:
                        continue
                    y_top_i = float(series_ann_i.max())
                    display_name_i = method_to_display_name.get(method, method)
                    fig.add_annotation(
                        x=display_name_i,
                        y=y_top_i + y_pad_i,
                        text="<b>ours</b>",
                        showarrow=False,
                        xref=f"x{col_idx}",
                        yref=f"y{col_idx}",
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(size=10),
                    )
        # metrics plotted

        for cat in categories_all:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=OPTIM_TO_COLOUR.get(cat, "#1f77b4"), size=10),
                    name=cat,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        _height = 400
        _width = _height * 2
        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            height=_height,
            width=_width,
            margin=dict(l=0, r=0, b=0, t=20),
            # automargin=False,
            # title_standoff=1,
            # ticklabelposition="inside",
            legend=dict(
                x=0.48,
                y=0.95,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
        )
        fig.update_yaxes(title_standoff=1, row=1, col=1)
        fig.update_yaxes(title_standoff=1, row=1, col=2)

        # Panel labels a (left) and b (right)
        dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.48]
        dom2 = (
            fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.52, 1.0]
        )
        fig.add_annotation(
            x=dom1[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text="<b>a</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
        fig.add_annotation(
            x=dom2[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text="<b>b</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
        fig.write_image(save_path, width=_width, height=_height, scale=3)
        print(f"Saved\n {save_path}")

    def _get_best_method_by_mean(_df, metric_name, prefer="min"):
        # prefer: "min" for metrics where lower is better; "max" where higher is better
        if _df is None or len(_df) == 0 or metric_name not in _df.columns:
            return None
        try:
            means = _df.groupby("name")[metric_name].mean()
            means = means.dropna()
            if len(means) == 0:
                return None
            if prefer == "max":
                return str(means.idxmax())
            return str(means.idxmin())
        except Exception:
            return None

    # df = df.sort_values(
    #     by="name",
    #     key=lambda s: s.map({name: i for i, name in enumerate(DO_METHOD)}),
    # )
    # sort df by average number of steps (highest first)
    df = df.sort_values(by="steps", ascending=False)

    # for DFT init, add 5min to the wall time
    df.loc[df["name"] == "RFO-BFGS (DFT init)", "wall_time_s"] += 5 * 60

    # for metric in ["steps", "grad_calls", "hessian_calls", "wall_time_s"]:
    for metric in ["steps", "wall_time_s"]:
        # Also create interactive Plotly violin ordered by descending mean steps
        _plot_metric_violin_plotly(
            df.copy(),
            metric,
            os.path.join(plots_dir, f"{metric}_violin_plotly.png"),
        )
        if metric == "wall_time_s":
            # # Competitive methods only
            # _d_comp = df.copy()[df["name"].isin(COMPETATIVE_METHODS_WALL_TIME)]
            # if len(_d_comp) > 0:
            #     _plot_metric_violin_plotly(
            #         _d_comp,
            #         metric,
            #         os.path.join(plots_dir, f"{metric}_violin_plotly_competative.png"),
            #     )
            # # Combined 3-panel figure: Steps | Wall Time | Wall Time (Competitive)
            # _plot_metric_violin_plotly_triple(
            #     df.copy(),
            #     os.path.join(plots_dir, "steps_walltime_walltime_competative_plotly.png"),
            # )
            # Combined 2-panel figure: Steps | Wall Time (Competitive)
            _plot_metric_violin_plotly_double(
                df.copy(),
                os.path.join(plots_dir, "steps_walltime_competative_plotly.png"),
            )

    # Convergence rate per method
    if "converged" in df.columns:
        conv = (
            df.groupby("name")["converged"].mean().reset_index(name="convergence_rate")
        )
        # sort by human name:
        conv = conv.sort_values(
            by="name",
            key=lambda s: s.map({name: i for i, name in enumerate(DO_METHOD)}),
        )
        # explicit order for x based on DO_METHOD
        order = [name for name in DO_METHOD if name in conv["name"].unique()]
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=conv,
            x="name",
            y="convergence_rate",
            order=order,
            # palette="tab10",
            # hue="coord",
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Convergence rate")
        ax.set_title(f"Convergence rate ({COORD_TO_NAME[args.coord]})")
        plt.xticks(rotation=25, ha="right")
        # # Highlight best method for convergence (highest mean converged), allow CLI override
        # best_conv = None
        # if len(conv) > 0 and "convergence_rate" in conv.columns:
        #     try:
        #         best_conv = str(conv.loc[conv["convergence_rate"].idxmax(), "name"])
        #     except Exception:
        #         best_conv = None
        # chosen = args.highlight_method if args.highlight_method else best_conv
        # if chosen is not None:
        #     for lbl in ax.get_xticklabels():
        #         if lbl.get_text() == chosen:
        #             lbl.set_fontweight("bold")
        plt.tight_layout()
        fname = os.path.join(plots_dir, "convergence_rate.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved\n {fname}")

    return df


def main():
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
    ap.add_argument("--max_samples", type=int, default=80)
    ap.add_argument("--max_cycles", type=int, default=150)
    ap.add_argument("--redo", type=bool, default=False)
    ap.add_argument("--thresh", type=str, default="gau")
    args = ap.parse_args()

    # Determine source label for logging
    source_label = os.path.splitext(args.xyz.split("/")[-1])[0]
    out_dir = os.path.join(
        ROOT_DIR,
        "runs_relaxation",
        source_label
        + "_"
        + "lossablation"
        + "_"
        + args.coord
        + "_"
        + args.thresh.replace("_", "")
        + "_"
        + str(args.max_samples),
    )

    df = do_relaxations(out_dir, source_label, args)
    plot_results(df, out_dir, args)


if __name__ == "__main__":
    main()
