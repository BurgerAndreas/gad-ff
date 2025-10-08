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
import shutil

import torch
# from torch_geometric.data import DataLoader as TGDataLoader

try:
    from pysisyphus.Geometry import Geometry  # Geometry API + coordinate systems
    from pysisyphus.calculators.MLFF import MLFF
    from pysisyphus.calculators.Calculator import (
        Calculator,
    )  # base class to wrap/override
    from pysisyphus.optimizers.FIRE import FIRE  # first-order baseline
    from pysisyphus.optimizers.RFOptimizer import RFOptimizer  # second-order RFO + BFGS
    from pysisyphus.optimizers.BFGS import BFGS
    from pysisyphus.optimizers.SteepestDescent import SteepestDescent
    from pysisyphus.optimizers.ConjugateGradient import ConjugateGradient
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


# from ase import Atoms
from ase.io import read
# from ase.calculators.emt import EMT
# from ase.constraints import FixAtoms
# from ase.vibrations.data import VibrationsData
# from ase.vibrations import Vibrations
# from ase.optimize import BFGS
# from ase.mep import NEB
# from sella import Sella, Constraints, IRC

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path, ROOT_DIR
from nets.prediction_utils import (
    GLOBAL_ATOM_SYMBOLS,
    GLOBAL_ATOM_NUMBERS,
    compute_extra_props,
)

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib

from ase.visualize.plot import plot_atoms

import pymol
from pymol import cmd
from PIL import Image

# Launch PyMOL once per process in headless mode
pymol.finish_launching(["pymol", "-c"])

import py3Dmol
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageSequence
import base64
import io


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


def save_and_animate_traj(
    trajectory, atoms, out_dir_method, idx, hessians=None, key="forces"
):
    traj_path = os.path.join(out_dir_method, f"traj_{idx}.xyz")
    save_trajectory_xyz(trajectory, atoms, traj_path, key=key)
    frames = [entry for entry in trajectory if entry["call"] == key]
    # if len(energy_frames) > 148:
    #     print(f"Skipping animation, too many frames ({len(energy_frames)})")
    #     return
    if len(frames) < 5:
        print(f"Skipping animation, too few frames ({len(frames)})")
        # Delete .h5 files
        for h5_file in glob.glob(os.path.join(out_dir_method, "*.h5")):
            os.remove(h5_file)
        return
    # # ase
    # gif_path = os.path.join(out_dir_method, f"traj_mpl_{idx}.gif")
    # animate_xyz_trajectory_ase(traj_path, gif_path)
    # #py3dmol, only works locally
    # gif_path = os.path.join(out_dir_method, f"traj_py3dmol_{idx}.gif")
    # animate_xyz_trajectory_py3dmol(traj_path, gif_path)
    # pymol
    gif_path = os.path.join(out_dir_method, f"traj_pymol_{idx}.gif")
    animate_xyz_trajectory_pymol(traj_path, gif_path)
    # Hessian
    if hessians is not None:
        hessian_gif_path = os.path.join(out_dir_method, f"hessian_anim_{idx}.gif")
        animate_hessian_trajectory(hessians, hessian_gif_path)
        # Combine side-by-side if frame counts match
        combined_gif_path = os.path.join(
            out_dir_method, f"combined_pymol_hessian_{idx}.gif"
        )
        combine_gifs_side_by_side(gif_path, hessian_gif_path, combined_gif_path)
    # Delete .h5 files
    for h5_file in glob.glob(os.path.join(out_dir_method, "*.h5")):
        os.remove(h5_file)


def save_trajectory_xyz(trajectory, atoms, filename, key="forces"):
    """Save trajectory from energy calls to XYZ file.

    Args:
        trajectory: list of dicts with 'call', 'coords', etc.
        atoms: list of element symbols
        filename: output XYZ path
        key: type of call to filter for (default 'forces')
    """
    energy_frames = [entry for entry in trajectory if entry["call"] == key]
    if len(energy_frames) == 0:
        print(f"No '{key}' frames found in trajectory")
        return

    n_atoms = len(atoms)
    with open(filename, "w") as f:
        for i, frame in enumerate(energy_frames):
            coords_bohr = frame["coords"]
            coords_ang = coords_bohr * BOHR2ANG
            coords_3d = coords_ang.reshape(n_atoms, 3)
            energy = frame.get("energy", None)

            f.write(f"{n_atoms}\n")
            if energy is not None:
                f.write(f"Frame {i}, Energy: {energy}\n")
            else:
                f.write(f"Frame {i}\n")

            for atom, xyz in zip(atoms, coords_3d):
                f.write(f"{atom:2s} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}\n")

    print(f"Saved {len(energy_frames)} frames to XYZ:\n {filename}")


def animate_xyz_trajectory_ase(
    xyz_file, output_gif, max_frames=100, rotation="10x,10y", figsize=(8, 8)
):
    """Create animated GIF of molecular trajectory using ASE.

    Args:
        xyz_file: path to XYZ trajectory file
        output_gif: path to save GIF file
        max_frames: maximum number of frames to include
        rotation: rotation string for plot_atoms (e.g., '10x,10y')
        figsize: figure size tuple
    """
    if not os.path.exists(xyz_file):
        print(f"XYZ file not found: {xyz_file}")
        return

    # Read all frames from XYZ file
    atoms_list = read(xyz_file, index=":")

    if len(atoms_list) == 0:
        print("No frames found in XYZ file")
        return

    # Subsample if too many frames
    step = max(1, len(atoms_list) // max_frames)
    atoms_list = atoms_list[::step]

    # print(f"Animating {len(atoms_list)} frames to {output_gif}")

    fig = plt.figure(figsize=figsize)

    def update(i):
        plt.clf()
        ax = fig.add_subplot(111)
        plot_atoms(atoms_list[i], ax, rotation=rotation, radii=0.5, show_unit_cell=0)
        ax.set_title(f"Frame {i}/{len(atoms_list) - 1}", fontsize=14)
        ax.axis("off")

    anim = FuncAnimation(fig, update, frames=len(atoms_list), interval=200, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=2)
    anim.save(output_gif, writer=writer)
    plt.close(fig)

    print(f"Saved matplotlib trajectory animation to\n {output_gif}")


def animate_xyz_trajectory_py3dmol(
    xyz_file, output_gif, max_frames=100, width=800, height=600
):
    """Create animated GIF of molecular trajectory using py3Dmol.

    Requires: pip install py3Dmol selenium pillow

    Args:
        xyz_file: path to XYZ trajectory file
        output_gif: path to save GIF file
        max_frames: maximum number of frames to include
        width: image width
        height: image height
    """
    if not os.path.exists(xyz_file):
        print(f"XYZ file not found: {xyz_file}")
        return

    atoms_list = read(xyz_file, index=":")

    if len(atoms_list) == 0:
        print("No frames found in XYZ file")
        return

    # Subsample if too many frames
    step = max(1, len(atoms_list) // max_frames)
    atoms_list = atoms_list[::step]

    # Setup headless browser
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    frames = []

    for i, atoms in enumerate(atoms_list):
        # Convert atoms to XYZ string
        xyz_str = f"{len(atoms)}\nFrame {i}\n"
        for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            xyz_str += f"{symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n"

        # Create py3Dmol view
        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_str, "xyz")
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}})
        view.setBackgroundColor("white")
        view.zoomTo()

        # Render to PNG (requires browser automation)
        try:
            png_data = view.png()
            img = Image.open(io.BytesIO(base64.b64decode(png_data)))
            frames.append(img)
        except (AssertionError, Exception) as e:
            print(f"py3Dmol rendering failed: {e}")
            return

        if (i + 1) % 10 == 0:
            print(f"Rendered {i + 1}/{len(atoms_list)} frames")

    # Save as GIF
    frames[0].save(
        output_gif, save_all=True, append_images=frames[1:], duration=400, loop=0
    )

    print(f"Saved py3Dmol trajectory animation to\n {output_gif}")


def animate_xyz_trajectory_pymol(
    xyz_file, output_gif, max_frames=100, width=800, height=800, ray_trace=False
):
    """Create animated GIF of molecular trajectory using PyMOL.

    Requires: PyMOL with Python API (conda install -c conda-forge pymol-open-source)

    Args:
        xyz_file: path to XYZ trajectory file
        output_gif: path to save GIF file
        max_frames: maximum number of frames to include
        width: image width
        height: image height
        ray_trace: if True, use ray tracing for higher quality (slower)
    """

    if not os.path.exists(xyz_file):
        print(f"XYZ file not found: {xyz_file}")
        return


    atoms_list = read(xyz_file, index=":")

    if len(atoms_list) == 0:
        print("No frames found in XYZ file")
        return

    n_frames = len(atoms_list)
    step = max(1, n_frames // max_frames)
    frame_indices = list(range(0, n_frames, step))
    
    print(f"Creating PyMOL animation: {n_frames} frames in XYZ → {len(frame_indices)} frames in GIF")

    # Reset PyMOL state for a fresh session
    cmd.reinitialize()

    # Configure PyMOL settings
    cmd.set("ray_opaque_background", 1)
    cmd.set("opaque_background", 1)
    cmd.bg_color("white")
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 0)
    cmd.set("connect_mode", 4)  # distance-based bonding guess
    # Flat, no-shadow style by default
    cmd.set("ray_shadows", 0)
    cmd.set("specular", 0) # no specular highlights (white spots)
    # cmd.set("ambient", 1) # completly flat
    cmd.set("depth_cue", 0) # further away does not become grayscaled
    # cmd.set("light_count", 1) # 1 to 10
    # Pastel element colors
    cmd.set_color("pastel_H", [240,240,240]) # #fafafa
    cmd.set_color("pastel_C", [0,142,109]) # #008b6b
    # cmd.set_color("pastel_N", [255,0,0]) # #006bf7
    cmd.set_color("pastel_N", [0,107,247]) # #006bf7
    cmd.set_color("pastel_O", [217,2,0]) # #ff0000

    # Render frames with per-frame bond regeneration by reloading single-state XYZ
    frames = []
    temp_dir = os.path.join(os.path.dirname(output_gif), ".pymol_temp")
    os.makedirs(temp_dir, exist_ok=True)

    saved_view = None

    for i, frame_idx in enumerate(frame_indices):
        atoms = atoms_list[frame_idx]

        # Write single-frame XYZ to temp file
        temp_xyz = os.path.join(temp_dir, f"frame_{i:04d}.xyz")
        with open(temp_xyz, "w") as fh:
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            fh.write(f"{len(symbols)}\nFrame {i}\n")
            for sym, pos in zip(symbols, positions):
                fh.write(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

        # cbaw: color by atom
        # oxygen in red, nitrogen in blue, hydrogen in white
        # carbon in white/grey
        # Load this frame as a fresh object so bonds are re-guessed
        if i == 0:
            cmd.load(temp_xyz, "traj")
            cmd.hide("everything", "traj")
            cmd.show("sticks", "traj")
            cmd.show("spheres", "traj")
            cmd.set("sphere_scale", 0.25)
            cmd.set("stick_radius", 0.15)
            # cmd.util.cbaw("traj")
            cmd.color("pastel_H", "traj and elem H")
            cmd.color("pastel_C", "traj and elem C")
            cmd.color("pastel_N", "traj and elem N")
            cmd.color("pastel_O", "traj and elem O")
            cmd.zoom("traj", buffer=0.5, complete=1)
            saved_view = cmd.get_view()
        else:
            cmd.delete("traj")
            cmd.load(temp_xyz, "traj")
            cmd.hide("everything", "traj")
            cmd.show("sticks", "traj")
            cmd.show("spheres", "traj")
            cmd.set("sphere_scale", 0.25)
            cmd.set("stick_radius", 0.15)
            # cmd.util.cbaw("traj")
            cmd.color("pastel_H", "traj and elem H")
            cmd.color("pastel_C", "traj and elem C")
            cmd.color("pastel_N", "traj and elem N")
            cmd.color("pastel_O", "traj and elem O")
            if saved_view is not None:
                cmd.set_view(saved_view)

        # Render frame
        temp_png = os.path.join(temp_dir, f"frame_{i:04d}.png")

        if ray_trace:
            cmd.ray(width, height)
            cmd.png(temp_png, width, height, dpi=300, ray=1)
        else:
            cmd.png(temp_png, width, height, dpi=150, ray=0)

        # Load image and ensure opaque (no alpha) to avoid GIF compositing trails
        img = Image.open(temp_png).convert("RGB")
        frames.append(img)

    # Clean up PyMOL
    cmd.delete("all")
    # pymol.cmd.quit()

    # Save as GIF
    if len(frames) > 3:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=400, # ms per frame
            loop=0,
            optimize=False,
            disposal=2,
        )
        print(f"PyMOL trajectory animation ({len(frames)} frames) to\n {output_gif}")

    # Clean up temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)


def animate_hessian_trajectory(hessian_records, output_path, max_frames=50):
    """Create animated GIF of Hessian evolution during optimization.

    Args:
        hessian_records: list of dicts with 'hessian', 'method', etc.
        output_path: path to save GIF file
        max_frames: maximum number of frames to include
    """
    if len(hessian_records) == 0:
        print("No Hessian records to animate")
        return

    # Subsample if too many frames
    # step = max(1, len(hessian_records) // max_frames)
    # frames = hessian_records[::step]
    frames = hessian_records

    # Determine global colorbar limits across all frames
    all_hessians = [f["hessian"] for f in frames if f["hessian"] is not None]
    if len(all_hessians) == 0:
        print("No valid Hessian data to animate")
        return

    vmin = min(h.min() for h in all_hessians)
    vmax = max(h.max() for h in all_hessians)

    # Create figure with single plot (fill canvas, no margins)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Create initial plot objects that will be updated
    im = None

    def update(frame_idx):
        nonlocal im
        ax.clear()

        record = frames[frame_idx]
        H = record["hessian"]
        method = record.get("method", "unknown")

        # Hessian heatmap
        im = ax.imshow(H, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")

        # Remove all axes and padding for borderless frames
        ax.set_xticks([])
        ax.set_yticks([])

        # plt.tight_layout(pad=0)
        ax.axis("off")

        return [im]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=2)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"Hessian animation ({len(frames)} frames) to\n {output_path}")


def count_gif_frames(gif_path):
    """Return the number of frames in a GIF."""
    if not os.path.exists(gif_path):
        return 0
    im = Image.open(gif_path)
    count = 0
    for _ in ImageSequence.Iterator(im):
        count += 1
    im.close()
    return count


def combine_gifs_side_by_side(gif_left_path, gif_right_path, output_path):
    """Combine two GIFs side by side after verifying equal frame counts.

    The left GIF is placed on the left, the right GIF on the right. If the two
    GIFs have different heights, the smaller one is vertically centered.
    """
    if (not os.path.exists(gif_left_path)) or (not os.path.exists(gif_right_path)):
        print("One or both GIF paths do not exist; skipping combine")
        return

    n_left = count_gif_frames(gif_left_path)
    n_right = count_gif_frames(gif_right_path)

    print(f"PyMOL frames: {n_left}, Hessian frames: {n_right}")

    if n_left == 0 or n_right == 0:
        print("One GIF has zero frames; skipping combine")
        return

    if n_left != n_right:
        if n_right == n_left + 1:
            print("Hessian has one extra frame; will ignore the last Hessian frame")
        else:
            print("Frame counts differ; not creating combined GIF")
            return

    left_im = Image.open(gif_left_path)
    right_im = Image.open(gif_right_path)

    # Determine a reasonable duration (ms) for output
    left_duration = left_im.info.get("duration") if isinstance(left_im.info, dict) else None
    right_duration = right_im.info.get("duration") if isinstance(right_im.info, dict) else None
    if left_duration is None and right_duration is None:
        duration = 400
    elif left_duration is None:
        duration = right_duration
    elif right_duration is None:
        duration = left_duration
    else:
        duration = max(left_duration, right_duration)

    frames_out = []

    left_iter = ImageSequence.Iterator(left_im)
    right_iter = ImageSequence.Iterator(right_im)

    for f_left, f_right in zip(left_iter, right_iter):
        fl = f_left.convert("RGB")
        fr = f_right.convert("RGB")
        h = max(fl.height, fr.height)
        w = fl.width + fr.width
        canvas = Image.new("RGB", (w, h), color="white")
        y1 = (h - fl.height) // 2
        y2 = (h - fr.height) // 2
        canvas.paste(fl, (0, y1))
        canvas.paste(fr, (fl.width, y2))
        frames_out.append(canvas)

    if len(frames_out) > 3:
        frames_out[0].save(
            output_path,
            save_all=True,
            append_images=frames_out[1:],
            duration=duration,
            loop=0,
            optimize=False,
            # PIL dispoasl=2 clears each frame to the background before showing the next one
            disposal=2,
        )
        print(f"combined GIF ({len(frames_out)} frames) to\n {output_path}")
    else:
        print(f"Skipping combined GIF, too few frames ({len(frames_out)})")
    left_im.close()
    right_im.close()


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

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.reset()

    def reset(self):
        self.energy_calls = 0
        self.grad_calls = 0
        self.hessian_calls = 0
        self.calculate_calls = 0
        self.calculate_energy_calls = 0
        self.calculate_gradient_calls = 0
        self.calculate_hessian_calls = 0
        # Trajectory & hessian histories
        self.trajectory = []  # list of dicts per call
        self.hessian_records = []  # list of dicts specifically for Hessians
        super().reset()

    @property
    def model(self):
        return self.inner.model

    # Delegate / count
    def get_energy(self, atoms, coords, **kw):
        self.energy_calls += 1
        energy = self.inner.get_energy(atoms, coords, **kw)
        self.trajectory.append(
            {
                "call": "energy",
                "coords": np.array(coords, copy=True),
                "energy": np.array(energy, copy=True) if energy is not None else None,
            }
        )
        return energy

    def get_forces(self, atoms, coords, **kw):
        self.grad_calls += 1
        forces = self.inner.get_forces(atoms, coords, **kw)
        self.trajectory.append(
            {
                "call": "forces",
                "coords": np.array(coords, copy=True),
                "forces": np.array(forces, copy=True) if forces is not None else None,
            }
        )
        return forces

    def get_hessian(self, atoms, coords, **kw):
        self.hessian_calls += 1
        results = self.inner.get_hessian(atoms, coords, **kw)
        method = getattr(self.inner, "hessian_method", None)
        # Extract Hessian from results dict if needed
        if isinstance(results, dict):
            hessian_array = results.get("hessian", None)
        else:
            hessian_array = results
        record = {
            "call": "hessian",
            "coords": np.array(coords, copy=True),
            "hessian": np.array(hessian_array, copy=True)
            if hessian_array is not None
            else None,
            "method": method,
        }
        self.trajectory.append(record)
        self.hessian_records.append(record)
        return results

    def get_num_hessian(self, atoms, coords, prepare_kwargs={}):
        self.hessian_calls += 1
        results = self.inner.get_num_hessian(atoms, coords, **prepare_kwargs)
        # Extract Hessian from results dict if needed
        if isinstance(results, dict):
            hessian_array = results.get("hessian", None)
        else:
            hessian_array = results
        record = {
            "call": "num_hessian",
            "coords": np.array(coords, copy=True),
            "hessian": np.array(hessian_array, copy=True)
            if hessian_array is not None
            else None,
            "method": "numerical",
        }
        self.trajectory.append(record)
        self.hessian_records.append(record)
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
                "wall_time_s": t1 - t0,
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
                "wall_time_s": None,
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
        out_dir=out_dir,
        max_cycles=max_cycles,
        gdiis=False,
        line_search=False,
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


def get_geom(atomssymbols, coords, coord_type, base_calc, args):
    geom = Geometry(atomssymbols, coords, coord_type=coord_type)
    base_calc.reset()
    counting_calc = CountingCalc(base_calc)
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
    # "NaiveSteepestDescent",
    # "SteepestDescent",
    # "FIRE",
    # "RFO (autograd)",
    # "RFO (NumHess)",
    # "ConjugateGradient",
    # "RFO-BFGS (unit init)",
    "RFO (learned)",
]


def do_relaxations(out_dir, source_label, args):
    print("Loading dataset...")
    print(f"Dataset: {args.xyz}. is file: {os.path.isfile(args.xyz)}")
    dataset_path = args.xyz
    print(f"Loading T1x dataset from {dataset_path}")
    dataset = T1xDFTDataloader(dataset_path, datasplit="train", only_final=True)

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

    rng = np.random.default_rng(seed=0)

    print("\nRunning relaxations...")
    csv_path = os.path.join(out_dir, f"relaxation_results.csv")
    if not os.path.exists(csv_path) or args.redo:
        print("\nInitializing model...")
        # base_calc = MLFF(
        base_calc = PysisEquiformer(
            charge=0,
            ckpt_path=args.ckpt_path,
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
        counting_calc = CountingCalc(base_calc)
        molecule = next(iter(dataset))
        if "positions_noised" in molecule[args.key]:
            coords = molecule[args.key]["positions_noised"]
        else:
            coords = molecule[args.key]["positions"]
        # ts = molecule["transition_state"]["positions"]
        # product = molecule["product"]["positions"]
        atoms = np.array(molecule[args.key]["atomic_numbers"])
        atomssymbols = [Z_TO_SYMBOL[a] for a in atoms]
        coords = coords / BOHR2ANG  # same as *ANG2BOHR
        t1xdataloader = iter(dataset)
        geom = Geometry(atomssymbols, coords, coord_type=args.coord)
        geom.set_calculator(counting_calc)
        energy = geom.energy
        forces = geom.forces
        hessian = geom.hessian
        # Test finished

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
            molecule = next(t1xdataloader)
            idx = molecule[args.key].get("idx", cnt)
            coords = molecule[args.key]["positions"]
            atoms = np.array(molecule[args.key]["atomic_numbers"])
            atomssymbols = [Z_TO_SYMBOL[a] for a in atoms]
            coords = coords / BOHR2ANG  # same as *ANG2BOHR
            # eV/Angstrom^2 -> Hartree/Bohr^2
            # initial_dft_hessian = initial_dft_hessian * AU2EV * BOHR2ANG * BOHR2ANG

            # # only keep larger molecules
            # natoms = len(atomssymbols)
            # if natoms < 10:
            #     continue

            print(
                "",
                "=" * 80,
                f"\tSample {optims_done} (tried cnt={cnt}, idx={idx} / {len_dataset})\t",
                "=" * 80,
                sep="\n",
            )

            if args.noiserms and args.noiserms > 0.0:
                # print(f"Adding noise to geometry with RMS {args.noiserms} Å")
                noise = rng.normal(0.0, 1.0, size=coords.shape)
                # Scale noise so RMS of per-atom Euclidean displacement equals noiserms
                current_rms = float(np.sqrt(np.mean(np.sum(noise * noise, axis=1))))
                scale = (args.noiserms / current_rms) if current_rms > 0.0 else 0.0
                displacement = scale * noise
                coords = coords + (displacement / BOHR2ANG)

            results = []

            # first order, with backtracking line search
            method_name = "SteepestDescent"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_sd = get_geom(atomssymbols, coords, args.coord, base_calc, args)
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = SteepestDescent(
                    geom_sd,
                    max_cycles=args.max_cycles,
                    thresh=args.thresh,
                    # line_search=True,
                    out_dir=out_dir_method,
                )
                result = _run_opt_safely(
                    geom=geom_sd,
                    opt=opt,
                    method_name=method_name,
                    out_dir=out_dir,
                    verbose=args.verbose,
                )
                results.append(result)
                save_and_animate_traj(
                    trajectory=geom_sd.calculator.trajectory,
                    atoms=atomssymbols,
                    out_dir_method=out_dir_method,
                    idx=idx,
                )

            # first-order optimization (FIRE)
            method_name = "FIRE"
            if method_name in DO_METHOD:
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                print_header(cnt, method_name)
                geom_fire = get_geom(atomssymbols, coords, args.coord, base_calc, args)
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
                    geom_fire,
                    max_cycles=args.max_cycles,
                    thresh=args.thresh,
                    # # Initial time step; adaptively scaled during optimization
                    # dt=0.1,
                    # # Maximum allowed time step when increasing dt
                    # dt_max=1,
                    # # Consecutive aligned steps before accelerating
                    # N_acc=2,
                    # # Factor to increase dt on acceleration
                    # f_inc=1.1,
                    # # Factor to reduce mixing a on acceleration; also shrinks dt on reset here
                    # f_acc=0.99,
                    # # Unused in this implementation; typical FIRE uses to reduce dt on reset
                    # f_dec=0.5,
                    # # Counter of aligned steps since last reset (start at 0)
                    # n_reset=0,
                    # # Initial mixing parameter for velocity/force mixing; restored on reset
                    # a_start=0.1,
                    # String poiting to a directory where optimization progress is
                    # dumped.
                    out_dir=out_dir_method,
                    allow_write=False,
                )
                result = _run_opt_safely(
                    geom=geom_fire,
                    opt=opt,
                    method_name=method_name,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                )
                results.append(result)
                save_and_animate_traj(
                    trajectory=geom_fire.calculator.trajectory,
                    atoms=atomssymbols,
                    out_dir_method=out_dir_method,
                    idx=idx,
                )

            # 2) No Hessian: BFGS with non-Hessian initial guess (unit) - pure quasi-Newton
            #    RFOptimizer accepts hessian_init and BFGS updates.
            method_name = "RFO-BFGS (unit init)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                # geom2 = Geometry(atomssymbols, coords, coord_type=args.coord)
                # geom2.set_calculator(CountingCalc(base_calc))
                geom_bfgsunit = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_bfgsunit,
                    hessian_init="unit",
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=out_dir_method,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                    allow_write=False,
                )
                result = _run_opt_safely(
                    geom=geom_bfgsunit,
                    opt=opt,
                    method_name=method_name,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                )
                results.append(result)
                save_and_animate_traj(
                    trajectory=geom_bfgsunit.calculator.trajectory,
                    atoms=atomssymbols,
                    out_dir_method=out_dir_method,
                    idx=idx,
                )

            # Periodic replace: k=1 (every step)
            method_name = "RFO (learned)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfolearned = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfolearned,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                result = _run_opt_safely(
                    geom=geom_rfolearned,
                    opt=opt,
                    method_name=method_name,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                )
                results.append(result)
                save_and_animate_traj(
                    trajectory=geom_rfolearned.calculator.trajectory,
                    hessians=geom_rfolearned.calculator.hessian_records,
                    atoms=atomssymbols,
                    out_dir_method=out_dir_method,
                    idx=idx,
                )

            # Periodic replace: k=1 (every step)
            method_name = "RFO (autograd)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                hessian_method_before = base_calc.hessian_method
                base_calc.hessian_method = "autograd"
                geom_rfoautograd = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfoautograd,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                result = _run_opt_safely(
                    geom=geom_rfoautograd,
                    opt=opt,
                    method_name=method_name,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                )
                results.append(result)
                save_and_animate_traj(
                    trajectory=geom_rfoautograd.calculator.trajectory,
                    hessians=geom_rfoautograd.calculator.hessian_records,
                    atoms=atomssymbols,
                    out_dir_method=out_dir_method,
                    idx=idx,
                )
                base_calc.hessian_method = hessian_method_before

            # Finite difference Hessian at every step
            method_name = "RFO (NumHess)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfonumhess = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                geom_rfonumhess.calculator.force_num_hessian()
                # numerical_hessian = geom_rfonumhess.calculator.get_num_hessian(geom_rfonumhess.atoms, geom_rfonumhess._coords)
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfonumhess,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                result = _run_opt_safely(
                    geom=geom_rfonumhess,
                    opt=opt,
                    method_name=method_name,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                )
                results.append(result)
                save_and_animate_traj(
                    trajectory=geom_rfonumhess.calculator.trajectory,
                    hessians=geom_rfonumhess.calculator.hessian_records,
                    atoms=atomssymbols,
                    out_dir_method=out_dir_method,
                    idx=idx,
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


def main():
    """
    uv run scripts/plot_relax_traj.py --redo True --thresh gau --noiserms 0.05
    uv run scripts/plot_relax_traj.py --redo True --thresh gau_loose --key ts
    uv run scripts/plot_relax_traj.py --redo True --thresh gau_loose --key ts --noiserms 0.0
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xyz",
        # default="ts1x-val.lmdb",
        default="/ssd/Code/Datastore/transition1x.h5",
        help="input geometry in form of .xyz or folder of .xyz files or lmdb or h5 path",
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
    ap.add_argument("--max_samples", type=int, default=25)
    ap.add_argument("--max_cycles", type=int, default=150)
    ap.add_argument("--debug", type=bool, default=False)
    ap.add_argument("--redo", type=bool, default=False)
    ap.add_argument("--verbose", type=bool, default=False)
    ap.add_argument("--thresh", type=str, default="gau")
    ap.add_argument("--key", type=str, default="r")
    ap.add_argument(
        "--noiserms",
        type=float,
        default=0.05,
        help="Per-atom RMS displacement (Å) added to geometry before Hessian; 0 disables noise",
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to ckpt file",
    )
    args = ap.parse_args()

    if args.ckpt_path is None:
        args.ckpt_path = "/ssd/Code/gad-ff/ckpt/hesspred_v1.ckpt"

    keymap = {
        "r": "reactant",
        "p": "product",
        "ts": "transition_state",
    }
    args.key = keymap[args.key]

    # Determine source label for logging
    source_label = os.path.splitext(args.xyz.split("/")[-1])[0]
    out_dir = os.path.join(
        ROOT_DIR,
        "runs_relaxation_plot",
        source_label
        + "_"
        + args.coord
        + "_"
        + args.thresh.replace("_", "")
        + "_"
        + str(args.max_samples)
        + "_"
        + args.key
        + "_"
        + str(args.noiserms),
    )

    df = do_relaxations(out_dir, source_label, args)


if __name__ == "__main__":
    main()
