from typing import List, Optional, Tuple

import os
import numpy as np
import scipy.linalg
import yaml
from tqdm import tqdm

import torch
import torch.nn
import torch.utils.data

from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
from ocpmodels.ff_lmdb import LmdbDataset
from nets.prediction_utils import compute_extra_props, GLOBAL_ATOM_NUMBERS
from gadff.path_config import DATASET_FILES_HORM, fix_horm_dataset_path

from rdkit.Chem import GetPeriodicTable

from ase import Atoms
from ase.vibrations.data import VibrationsData
from ase.units import Hartree, Bohr

import geometric.molecule 
import geometric.normal_modes 
from geometric.nifty import au2kj, bohr2ang, c_lightspeed

"""
get the eigenvectors of the hessian of a force field, 
that do not correspond to extra rotation or translation degrees of freedom (invariance of the energy)
"""

# helper functions

def _is_linear_molecule(coords, threshold=1e-8):
    """
    Check if a molecule is linear by examining the geometry
    
    Args:
        coords: numpy array of shape (N, 3) with atomic coordinates
        threshold: tolerance for linearity detection
    
    Returns:
        bool: True if molecule is linear
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    N = len(coords)
    if N <= 2:
        return True
    
    # Center coordinates
    com = np.mean(coords, axis=0)
    coords_centered = coords - com
    
    # Compute inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for i in range(N):
        r = coords_centered[i]
        inertia_tensor += np.outer(r, r)
    
    # Check if smallest eigenvalue is much smaller than others
    eigenvals = np.linalg.eigvals(inertia_tensor)
    eigenvals = np.sort(eigenvals)
    
    # Linear if smallest eigenvalue is much smaller than largest
    return eigenvals[0] < threshold * eigenvals[-1]


def _get_masses(atom_types):
    pt = GetPeriodicTable()
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.tolist()
    if isinstance(atom_types[0], str):
        atom_types = [pt.GetAtomicNumber(z) for z in atom_types]
    masses = torch.tensor(
        [pt.GetAtomicWeight(z) for z in atom_types], dtype=pos.dtype, device=pos.device
    )
    return masses

def _compute_numerical_rank_threshold(
    evals: torch.Tensor, matrix_shape: tuple
) -> float:
    """
    Compute adaptive threshold for numerical rank determination using the same algorithm as
    NumPy's matrix_rank and MATLAB's rank function.

    This sets threshold = max(singular_values) * max(matrix_dimensions) * machine_epsilon

    Args:
        evals: Eigenvalues (can be negative for Hessian)
        matrix_shape: Shape of the matrix (M, N)

    Returns:
        Adaptive threshold value
    """
    abs_evals = torch.abs(evals)
    max_eval = torch.max(abs_evals)

    # Use the same algorithm as NumPy's matrix_rank: S.max() * max(M, N) * eps
    M, N = matrix_shape
    eps = torch.finfo(evals.dtype).eps
    threshold = max_eval * max(M, N) * eps

    return threshold.item()


def _get_ndrop(evals, m_shape, threshold, sorted_evals=None, islinear=False, print_warning=False):
    expected_dof = 5 if islinear else 6
    if sorted_evals is None:
        sorted_evals, sort_idx = torch.sort(torch.abs(evals))
    # Compute adaptive threshold using NumPy's matrix_rank algorithm
    if threshold == "auto":
        threshold = _compute_numerical_rank_threshold(evals, m_shape)
        ndrop = (sorted_evals < threshold).sum().item()
    elif isinstance(threshold, float):
        threshold = threshold
        ndrop = (sorted_evals < threshold).sum().item()
    elif threshold is None:
        ndrop = expected_dof
    else:
        raise ValueError("threshold must be 'auto' or float")
    if print_warning and ndrop != expected_dof:
        print("W: Error in projector-based removal of translations & rotations")
        print(f"W: Num eigenvalues below threshold: {ndrop}, should be {expected_dof}")
        print(f"W: Threshold: {threshold:.2e}")
        print(sorted_evals[:8])
        ndrop = expected_dof
    return threshold, ndrop


# main functions to test


def get_modes_geometric(hessian, coords, atom_types, return_raw_eigenvalues=False, unmass_weight=False):
    """
    Use Geometric for frequency analysis

    Args:
        hessian: torch.Tensor Hessian matrix
        coords: torch.Tensor coordinates in Angstrom
        atom_types: list of atomic numbers
        return_raw_eigenvalues: bool, if True return eigenvalues in atomic units
                               instead of frequencies in cm⁻¹
        unmass_weight: bool, if True and return_raw_eigenvalues=True, return raw Cartesian 
                      Hessian eigenvalues (Hartree/Bohr²) instead of mass-weighted ones

    Returns:
        If return_raw_eigenvalues=False (default):
            freqs_torch: frequencies in cm⁻¹
            modes_torch: normal modes
        If return_raw_eigenvalues=True and unmass_weight=False:
            eigenvals_torch: mass-weighted Hessian eigenvalues in atomic units (Hartree/amu)
            modes_torch: normal modes
        If return_raw_eigenvalues=True and unmass_weight=True:
            eigenvals_torch: raw Cartesian Hessian eigenvalues in atomic units (Hartree/Bohr²)
            modes_torch: normal modes (from mass-weighted calculation)
    
    Usage:
        # 1. Default: frequencies in cm⁻¹
        freqs_cm, modes = get_modes_geometric(hessian, coords, atom_types)

        # 2. Mass-weighted eigenvalues in atomic units (Hartree/amu)
        eigenvals_mw, modes = get_modes_geometric(hessian, coords, atom_types, return_raw_eigenvalues=True)

        # 3. Raw Cartesian eigenvalues in atomic units (Hartree/Bohr²)
        eigenvals_cart, modes = get_modes_geometric(hessian, coords, atom_types, return_raw_eigenvalues=True, unmass_weight=True)
    """

    # Convert to numpy
    if isinstance(hessian, torch.Tensor):
        hessian = hessian.detach().cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    # Convert atomic numbers to symbols
    elements = [GetPeriodicTable().GetElementSymbol(z) for z in atom_types]
    
    # Convert coordinates from Angstrom to Bohr (geometric expects Bohr)
    coords_bohr = coords.flatten() / Bohr  # Convert Å to Bohr

    # Get the normal modes. use geometric for proper removal of translational/rotational modes
    freqs, modes, G_tot_au = geometric.normal_modes.frequency_analysis(coords_bohr, hessian, elem=elements, verbose=False)
    
    if return_raw_eigenvalues and unmass_weight:
        # Now diagonalize the raw Cartesian Hessian to get unweighted eigenvalues
        # Remove TR modes using the same projection as geometric would use
        na = len(atom_types)
        coords_2d = coords_bohr.reshape(-1, 3)
        
        # Get masses for the projection (but don't use them for weighting)
        masses = np.array([GetPeriodicTable().GetAtomicWeight(z) for z in atom_types])
        
        # Create projection matrix to remove TR modes (simplified version of geometric's approach)
        # This is a basic implementation - for production use, should use geometric's exact projection
        from scipy.linalg import eigh
        
        # Mass-weight for projection only
        invsqrtm3 = 1.0/np.sqrt(np.repeat(masses, 3))
        wHessian_proj = hessian.copy() * np.outer(invsqrtm3, invsqrtm3)
        
        # Quick eigendecomposition to get the vibrational subspace
        all_evals, all_evecs = eigh(wHessian_proj)
        
        # Keep the same number of modes as geometric returns
        n_vib_modes = len(freqs)
        # Sort by absolute value and take the largest n_vib_modes
        sorted_idx = np.argsort(np.abs(all_evals))
        vib_idx = sorted_idx[-n_vib_modes:]
        
        # Project the unweighted Hessian into the vibrational subspace
        vib_evecs_mw = all_evecs[:, vib_idx]
        vib_evecs_cart = vib_evecs_mw / invsqrtm3[:, np.newaxis]
        
        # Get eigenvalues of the raw Hessian in the vibrational subspace
        H_vib = vib_evecs_cart.T @ hessian @ vib_evecs_cart
        eigenvals_cart, _ = eigh(H_vib)
        
        # Convert back to torch
        eigenvals_torch = torch.from_numpy(eigenvals_cart)
        modes_torch = torch.from_numpy(modes)
        
        return eigenvals_torch, modes_torch
    
    else:

        if return_raw_eigenvalues:
            # Convert frequencies back to mass-weighted eigenvalues in atomic units
            # Reverse the conversion done in frequency_analysis:
            # mwHess_wavenumber = 1e10*np.sqrt(au2kj / bohr2nm**2)/(2*np.pi*c_lightspeed)
            # freqs_wavenumber = mwHess_wavenumber * np.sqrt(np.abs(ichess_vals)) * np.sign(freqs_wavenumber)
            
            bohr2nm = bohr2ang / 10
            mwHess_wavenumber = 1e10*np.sqrt(au2kj / bohr2nm**2)/(2*np.pi*c_lightspeed)
            
            # Reverse conversion: ichess_vals = (freqs_wavenumber / mwHess_wavenumber)^2 * sign(freqs_wavenumber)
            eigenvals = (freqs / mwHess_wavenumber)**2 * np.sign(freqs)
            
            # Convert back to torch
            eigenvals_torch = torch.from_numpy(eigenvals)
            modes_torch = torch.from_numpy(modes)
            
            return eigenvals_torch, modes_torch
        else:
            # Convert back to torch
            freqs_torch = torch.from_numpy(freqs)
            modes_torch = torch.from_numpy(modes)

            return freqs_torch, modes_torch


def get_vibrational_modes_ase(hessian_torch, coords, atom_types, debug=False):
    """
    Use ASE to get vibrational modes from torch Hessian
    
    Intelligently separates translational/rotational modes from vibrational modes:
    1. If exactly 6 (or 5 for linear) imaginary freqs: treat as TR modes
    2. If more imaginary freqs: smallest are TR, extras are transition state modes  
    3. If fewer imaginary freqs: supplement with lowest real frequencies

    Args:
        hessian_torch: torch.Tensor of shape (3*N, 3*N) - Hessian matrix
        coords: torch.Tensor of shape (N, 3) - atomic coordinates
        atom_types: list of atomic numbers
        debug: bool - whether to print diagnostic information

    Returns:
        frequencies_vib: frequencies in cm^-1 for vibrational modes only (3N-6 or 3N-5)
        modes_vib: vibrational modes excluding TR modes
        frequencies_all: all frequencies including TR modes (for analysis)
        n_tr_modes: number of translational/rotational modes detected
    """

    # Convert to numpy
    hessian_np = hessian_torch.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()
    masses = _get_masses(atom_types)
    masses_np = masses.detach().cpu().numpy()
    symbols = [GetPeriodicTable().GetElementSymbol(z) for z in atom_types]

    # Create ASE atoms object
    atoms = Atoms(symbols=symbols, positions=coords_np, masses=masses_np)

    # Create VibrationsData object
    vib_data = VibrationsData.from_2d(atoms, hessian_np)

    # Get all frequencies and modes
    frequencies_all = vib_data.get_frequencies()  # in cm^-1 (complex for imaginary)
    modes_all = vib_data.get_modes()  # shape (3*N, N, 3)
    
    # Convert complex frequencies to real (negative for imaginary)
    frequencies_real = np.where(
        np.isreal(frequencies_all), 
        frequencies_all.real, 
        -np.abs(frequencies_all.imag)
    )
    
    # Check for linearity
    is_linear = _is_linear_molecule(coords_np)
    n_tr_expected = 5 if is_linear else 6  # Linear: 3 trans + 2 rot, Non-linear: 3 trans + 3 rot
    
    # Separate imaginary and real frequencies
    is_imaginary = ~np.isreal(frequencies_all)
    imaginary_indices = np.where(is_imaginary)[0]
    real_indices = np.where(~is_imaginary)[0]
    
    n_imaginary = len(imaginary_indices)
    
    # Strategy: 
    # 1. If we have exactly the expected number of imaginary freqs, assume they are TR modes
    # 2. If we have more, assume the extra ones are transition state modes (keep them as vibrational)
    # 3. If we have fewer, supplement with lowest real frequencies
    
    if n_imaginary != n_tr_expected and debug:
        print(f"Imaginary frequencies: {n_imaginary}")
        print(f"Expected TR modes: {n_tr_expected}")
        print(f"Molecule is {'linear' if is_linear else 'non-linear'}")
        print(f"frequencies_all\n", np.sort(np.abs(frequencies_all))[:10])
        print(f"frequencies_real\n", np.sort(np.abs(frequencies_real))[:10])
        print(f"frequencies_all[imaginary_indices]\n", frequencies_all[imaginary_indices][:10])
    
    if n_imaginary == n_tr_expected:
        # Perfect case: imaginary frequencies are exactly the TR modes
        tr_indices = imaginary_indices
        vib_indices = real_indices
        
    elif n_imaginary > n_tr_expected:
        # Extra imaginary frequencies - likely transition state
        # Sort imaginary frequencies by absolute value (smallest = most likely TR)
        imag_abs_vals = np.abs(frequencies_all[imaginary_indices].imag)
        imag_sorted_idx = np.argsort(imag_abs_vals)
        
        # Take the smallest (absolute value) imaginary frequencies as TR modes
        tr_from_imag = imaginary_indices[imag_sorted_idx[:n_tr_expected]]
        extra_imag = imaginary_indices[imag_sorted_idx[n_tr_expected:]]
        
        tr_indices = tr_from_imag
        vib_indices = np.concatenate([extra_imag, real_indices])
        
    else:
        # Fewer imaginary frequencies than expected TR modes
        # Use all imaginary + lowest real frequencies to complete TR modes
        n_real_needed = n_tr_expected - n_imaginary
        
        if n_real_needed > 0:
            # Sort real frequencies by absolute value
            real_abs_vals = np.abs(frequencies_real[real_indices])
            real_sorted_idx = np.argsort(real_abs_vals)
            
            tr_from_real = real_indices[real_sorted_idx[:n_real_needed]]
            remaining_real = real_indices[real_sorted_idx[n_real_needed:]]
            
            tr_indices = np.concatenate([imaginary_indices, tr_from_real])
            vib_indices = remaining_real
        else:
            tr_indices = imaginary_indices
            vib_indices = real_indices
            
    # Sort indices for consistent output
    tr_indices = np.sort(tr_indices)
    vib_indices = np.sort(vib_indices)
    
    # Extract vibrational modes
    frequencies_vib = frequencies_real[vib_indices]
    modes_vib = modes_all[vib_indices]  # shape (n_vib, N, 3)
    
    # Convert back to torch
    frequencies_vib_torch = torch.from_numpy(frequencies_vib).to(hessian_torch.device)
    modes_vib_torch = torch.from_numpy(modes_vib).to(hessian_torch.device)
    frequencies_all_torch = torch.from_numpy(frequencies_real).to(hessian_torch.device)
    
    return frequencies_vib_torch, modes_vib_torch, frequencies_all_torch, len(tr_indices)


def compute_internal_modes(
    hessian, atom_types, coords, threshold_linear=1e-8, threshold_internal=5e-4
):
    """
    Compute eigenvalues/eigenvectors of Hessian excluding translational/rotational modes.
    Automatically detects linear molecules.

    Args:
        hessian (np.ndarray): Cartesian Hessian (3N x 3N)
        masses (np.ndarray): Atomic masses (N,)
        coords (np.ndarray): Atomic coordinates (N x 3)
        threshold_linear (float): Numerical tolerance for linearity detection
        threshold_internal (float): Numerical tolerance for internal mode detection

    Returns:
        internal_evals (np.ndarray): Eigenvalues of internal modes (3N-6 or 3N-5)
        internal_evecs_cart (np.ndarray): Cartesian eigenvectors (3N x (3N-6) or (3N x (3N-5))
        internal_evecs_mw (np.ndarray): Mass-weighted eigenvectors (same shape as above)
        rank_ext (int): Number of external modes detected (3, 5, or 6)
    """
    
    N = len(atom_types)
    total_dof = 3 * N
    
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(hessian, torch.Tensor):
        hessian = hessian.detach().cpu().numpy()

    # Center coordinates at center of mass
    masses = _get_masses(atom_types)
    if isinstance(masses, torch.Tensor):
        masses = masses.detach().cpu().numpy()
    com = np.sum(masses[:, None] * coords, axis=0) / np.sum(masses)
    coords_rel = coords - com[None, :]

    # 1. Detect linearity using inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for i in range(N):
        x, y, z = coords_rel[i]
        m = masses[i]
        inertia_tensor[0, 0] += m * (y**2 + z**2)
        inertia_tensor[1, 1] += m * (x**2 + z**2)
        inertia_tensor[2, 2] += m * (x**2 + y**2)
        inertia_tensor[0, 1] -= m * x * y
        inertia_tensor[0, 2] -= m * x * z
        inertia_tensor[1, 2] -= m * y * z
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    # Compute eigenvalues of inertia tensor
    inertia_eigvals = np.linalg.eigvalsh(inertia_tensor)
    is_linear = inertia_eigvals[0] < threshold_linear * max(1, np.max(inertia_eigvals))

    # Determine number of external modes
    if N == 1:  # Single atom
        rank_ext = 3
    elif is_linear:
        rank_ext = 5  # Linear molecule: 3 trans + 2 rot
        print("W: Linear molecule detected")
    else:
        rank_ext = 6  # Nonlinear molecule: 3 trans + 3 rot

    # 2. Mass-weight the Hessian
    sqrt_masses_rep = np.repeat(np.sqrt(masses), 3)
    inv_sqrt_masses_rep = 1.0 / sqrt_masses_rep
    H_mw = inv_sqrt_masses_rep[:, None] * hessian * inv_sqrt_masses_rep[None, :]

    # 3. Build external modes matrix B (3N x 6)
    B = np.zeros((total_dof, 6))
    # Translations (always present)
    B[0::3, 0] = np.sqrt(masses)  # X
    B[1::3, 1] = np.sqrt(masses)  # Y
    B[2::3, 2] = np.sqrt(masses)  # Z

    # Rotations (skip if single atom)
    if N > 1:
        # Reshape relative coordinates to vector [x0,y0,z0, x1,y1,z1, ...]
        coords_vec = coords_rel.reshape(-1)
        B[1::3, 3] = -coords_vec[2::3] * np.sqrt(masses)  # -z_i * √m_i
        B[2::3, 3] = coords_vec[1::3] * np.sqrt(masses)  #  y_i * √m_i
        B[0::3, 4] = coords_vec[2::3] * np.sqrt(masses)  #  z_i * √m_i
        B[2::3, 4] = -coords_vec[0::3] * np.sqrt(masses)  # -x_i * √m_i
        B[0::3, 5] = -coords_vec[1::3] * np.sqrt(masses)  # -y_i * √m_i
        B[1::3, 5] = coords_vec[0::3] * np.sqrt(masses)  #  x_i * √m_i

    # 4. Orthonormalize external modes (using first `rank_ext` columns)
    U, _, _ = np.linalg.svd(B[:, :rank_ext], full_matrices=False)
    U_ext = U[:, :rank_ext]  # Orthonormal basis for external modes

    # 5. Project Hessian into internal space
    projector_ext = U_ext @ U_ext.T
    projector_int = np.eye(total_dof) - projector_ext
    H_int = projector_int.T @ H_mw @ projector_int

    # 6. Diagonalize projected Hessian
    evals, evecs_mw = scipy.linalg.eigh(H_int)
    
    # sorted_evals_idx = np.argsort(np.abs(evals))
    # evals_sorted = evals[sorted_evals_idx]
    # print("evals_sorted\n", evals_sorted[:10])

    # 7. Filter internal modes (ignore near-zero eigenvalues)
    mask = np.abs(evals) > threshold_internal
    internal_evals = evals[mask]
    internal_evecs_mw = evecs_mw[:, mask]
    
    if mask.sum() != total_dof - rank_ext:
        print(f"Internal modes: {mask.sum()} != {total_dof - rank_ext}")
        print(f"evals_sorted\n", np.sort(np.abs(evals))[:10])
        print(f"threshold_internal={threshold_internal}")

    # 8. Convert to Cartesian eigenvectors
    internal_evecs_cart = internal_evecs_mw / sqrt_masses_rep[:, None]

    return internal_evals, internal_evecs_cart, internal_evecs_mw, rank_ext


def projector_vibrational_modes(pos, atom_types, H, threshold=5e-4):
    """
    Projector-based removal of translations & rotations
    A robust alternative to filtering is to build the subspace of exactly invariant motions
    (3 translations + 3 rotations, or 5 rotations for a linear molecule)
    and project them out of your Hessian before diagonalizing.
    This guarantees that no physical vibrational mode is accidentally thrown away.

    Args:
        pos: (N,3) positions;
        atom_types: list[int] of length N of atomic numbers;
        H: (3N,3N) Cartesian Hessian
    Returns:
        evals_vib: (3N-6,) eigenvalues of the vibrational modes, sorted in ascending order
        eigvecs_vib: (3N,3N-6) eigenvectors of the vibrational modes
    """
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.tolist()
    N = len(atom_types)
    # 0) Build mass vector m3 = [sqrt(m1), sqrt(m1), sqrt(m1), sqrt(m2), ...]
    masses = _get_masses(atom_types)
    sqrt_m3 = masses.repeat_interleave(3).sqrt()  # (3N,)

    # 1) mass-weight Hessian F = M^{-1/2} H M^{-1/2}
    F = H / (sqrt_m3[:, None] * sqrt_m3[None, :])  # (3N,3N)

    # 2) build T (3N×6) of rigid-body vectors
    # translations
    T = []
    for alpha in range(3):
        t = torch.zeros(3 * N, device=H.device, dtype=H.dtype)
        t[alpha::3] = sqrt_m3[alpha::3]  # x: indices 0,3,6...; y:1,4,7...; z:2,5,8...
        T.append(t)
    # rotations about COM
    com = (pos * masses[:, None]).sum(dim=0) / masses.sum()
    for axis in [0, 1, 2]:  # x,y,z rotation
        r = torch.zeros(3 * N, device=H.device, dtype=H.dtype)
        for i in range(N):
            x, y, z = pos[i] - com
            m3 = sqrt_m3[3 * i : 3 * i + 3]
            if axis == 0:  # rotate about x: (0, -z, y)
                r[3 * i : 3 * i + 3] = torch.tensor([0, -z, y], device=H.device) * m3
            elif axis == 1:  # about y: ( z, 0, -x)
                r[3 * i : 3 * i + 3] = torch.tensor([z, 0, -x], device=H.device) * m3
            else:  # about z: (-y, x, 0)
                r[3 * i : 3 * i + 3] = torch.tensor([-y, x, 0], device=H.device) * m3
        T.append(r)
    T = torch.stack(T, dim=1)  # (3N,6)

    # 3) orthonormalize T → U via QR
    Q, _ = torch.linalg.qr(T)  # Q: (3N,6) orthonormal
    # Build projector P = I - Q Q^T
    P = torch.eye(3 * N, device=H.device) - Q @ Q.T

    # 4) project the mass-weighted Hessian & diagonalize
    Fp = P @ (F @ P)
    evals, evecs = torch.linalg.eigh(Fp)  # mass-weighted vibrational modes

    # 5) drop the six exact zeros
    # sort by magnitude
    sorted_evals, sorted_evals_idx = torch.sort(torch.abs(evals))
    sorted_evecs = evecs[:, sorted_evals_idx]

    # 6) drop the six exact zeros, un-mass-weight the rest
    islinear = _is_linear_molecule(pos)
    threshold, ndrop = _get_ndrop(
        evals, Fp.shape, threshold, sorted_evals, islinear=islinear, print_warning=False
    )
    keep = torch.arange(ndrop, 3 * N)

    # The eigenvalues ω² in mass‐weighted Fp are the same for Cartesian H
    evals_vib = sorted_evals[keep]
    # Un‐mass‐weight to get Cartesian eigenvectors
    evecs_vib = sorted_evecs[:, keep] / sqrt_m3[:, None]

    # sort by smallest eigenvalues (same output as torch.linalg.eigh)
    sorted_evals_vib, sorted_evals_vib_idx = torch.sort(evals_vib)
    sorted_evecs_vib = evecs_vib[:, sorted_evals_vib_idx]

    return sorted_evals_vib, sorted_evecs_vib


def compute_cartesian_modes(pos, atom_types, H, orth_method="svd", threshold=None):
    """
    Compute vibrational eigenvalues and eigenvectors for a transition state Hessian H,
    removing translations and rotations via the Eckart frame and specified orthonormalization.

    Parameters
    ----------
    pos : torch.Tensor, shape (N, 3)
        Atomic positions.
    atom_types : list of str, length N
        Element symbols, e.g. ['C', 'H', 'H', ...].
    H : torch.Tensor, shape (3N, 3N)
        Cartesian Hessian (symmetric).
    orth_method : str, 'svd' or 'qr'
        Method to orthonormalize rigid-body vectors ('svd' for SVD, 'qr' for Gram–Schmidt/QR).
    threshold : float, optional
        Threshold for dropping modes. If None, drop 6 modes.

    Returns
    -------
    eigvals_cart : torch.Tensor, shape (3N - n_drop,)
        Cartesian Hessian eigenvalues (ω²), including one negative for TS.
    eigvecs_cart : torch.Tensor, shape (3N, 3N - n_drop)
        Corresponding Cartesian displacement eigenvectors (columns).
    """
    N = pos.shape[0]
    assert H.shape == (3 * N, 3 * N), "Hessian must be of shape (3N, 3N)"

    pt = GetPeriodicTable()
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.tolist()
    if isinstance(atom_types[0], str):
        atom_types = [pt.GetAtomicNumber(z) for z in atom_types]

    # 1. Compute masses and mass‐weighted positions
    masses = torch.tensor(
        [pt.GetAtomicWeight(z) for z in atom_types], dtype=pos.dtype, device=pos.device
    )
    m3 = masses.repeat_interleave(3).sqrt()

    # 2. Center on mass and align to principal axes (Eckart frame)
    com = (pos * masses[:, None]).sum(dim=0) / masses.sum()
    pos_centered = pos - com
    # Inertia tensor
    Idm = torch.zeros((3, 3), dtype=pos.dtype, device=pos.device)
    for i in range(N):
        r = pos_centered[i]
        m = masses[i]
        Idm += m * (r.dot(r) * torch.eye(3, device=pos.device) - torch.ger(r, r))
    # Principal axes
    eig_I, axes = torch.linalg.eigh(Idm)
    pos_eckart = (axes.T @ pos_centered.T).T  # rotate positions

    # 3. Mass‐weight Hessian
    F = H / (m3[:, None] * m3[None, :])

    # 4. Build rigid‐body vectors in mass‐weighted coordinates
    T_list = []
    # translations
    for alpha in range(3):
        t = torch.zeros(3 * N, dtype=pos.dtype, device=pos.device)
        t[alpha::3] = m3[alpha::3]
        T_list.append(t)
    # rotations about principal axes
    for axis in range(3):
        r = torch.zeros(3 * N, dtype=pos.dtype, device=pos.device)
        for i in range(N):
            x, y, z = pos_eckart[i]
            m3_i = m3[3 * i : 3 * i + 3]
            if axis == 0:
                vec = torch.tensor([0.0, -z, y], device=pos.device)
            elif axis == 1:
                vec = torch.tensor([z, 0.0, -x], device=pos.device)
            else:
                vec = torch.tensor([-y, x, 0.0], device=pos.device)
            r[3 * i : 3 * i + 3] = vec * m3_i
        T_list.append(r)
    T = torch.stack(T_list, dim=1)  # (3N, 6)

    # 5. Orthonormalize T to get Q
    if orth_method.lower() == "svd":
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)
        Q = U[:, :6]
    elif orth_method.lower() == "qr":
        Q, _ = torch.linalg.qr(T)
    else:
        raise ValueError("orth_method must be 'svd' or 'qr'")

    # 6. Projector P = I - Q Q^T
    I3N = torch.eye(3 * N, dtype=pos.dtype, device=pos.device)
    P = I3N - Q @ Q.T

    # 7. Project and diagonalize
    Fp = P @ (F @ P)
    evals, evecs = torch.linalg.eigh(Fp)

    # 8. Drop zero modes
    # sort by magnitude
    sorted_evals, sorted_evals_idx = torch.sort(torch.abs(evals))
    sorted_evecs = evecs[:, sorted_evals_idx]
    islinear = _is_linear_molecule(pos)
    threshold, ndrop = _get_ndrop(evals, Fp.shape, threshold, sorted_evals, islinear=islinear, print_warning=False)
    keep = torch.arange(ndrop, 3 * N)
    evals_vib = sorted_evals[keep]
    q_vib = sorted_evecs[:, keep]
    
    if ndrop != 6:
        print(f"Eckart {orth_method}: ndrop: {ndrop}")
        print(f"sorted_evals\n", sorted_evals[:10])
        print(f"threshold={threshold}")

    # 9. Un‐mass‐weight to Cartesian eigenvectors
    eigvecs_cart = q_vib / m3[:, None]
    eigvals_cart = evals_vib

    return eigvals_cart, eigvecs_cart


if __name__ == "__main__":

    project_root = os.path.dirname(os.path.dirname(__file__))

    # Example 1: load a dataset file and predict the first batch

    # dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    DATASET_FILES_HORM = [
        "ts1x-val.lmdb",  # 50844 samples
        "ts1x_hess_train_big.lmdb",  # 1725362 samples
        "RGD1.lmdb",  # 60000 samples
    ]
    dataset_path = fix_horm_dataset_path("RGD1.lmdb")
    dataset = LmdbDataset(dataset_path)

    max_samples = 100

    count_wrong_eckart_svd = 0
    count_wrong_eckart_qr = 0
    count_wrong_projection = 0
    count_wrong_internal = 0
    count_wrong_geometric = 0
    count_wrong_ase = 0
    
    print("\n")
    for i, sample in enumerate(dataset):
        if i >= max_samples and max_samples > 0:
            break

        indices = sample.one_hot.long().argmax(dim=1)
        sample.z = GLOBAL_ATOM_NUMBERS[indices]

        hessian = sample.hessian
        atom_types = sample.z.tolist()
        pos = sample.pos
        N = len(atom_types)

        num_vibrational_modes = 3 * N - 6

        hessian = hessian.reshape(3 * N, 3 * N)

        evals_vib, eigvecs_vib = projector_vibrational_modes(pos, atom_types, hessian)
        if len(evals_vib) != num_vibrational_modes:
            count_wrong_projection += 1
            diff = num_vibrational_modes - len(evals_vib)
            print(
                f"Projection {i}: vibrational modes: {diff}",
                "❌",
            )

        evals_vib_cart, eigvecs_vib_cart = compute_cartesian_modes(
            pos, atom_types, hessian
        )
        if len(evals_vib_cart) != num_vibrational_modes:
            count_wrong_eckart_svd += 1
            diff = num_vibrational_modes - len(evals_vib_cart)
            print(
                f"Eckart SVD {i}: vibrational modes: {diff}",
                "❌",
            )
        evals_vib_cart, eigvecs_vib_cart = compute_cartesian_modes(
            pos, atom_types, hessian, orth_method="qr"
        )
        if len(evals_vib_cart) != num_vibrational_modes:
            count_wrong_eckart_qr += 1
            diff = num_vibrational_modes - len(evals_vib_cart)
            print(
                f"Eckart QR {i}: vibrational modes: {diff}",
                "❌",
            )

        evals_internal, eigvecs_internal, eigvecs_internal_mw, rank_ext = compute_internal_modes(hessian, atom_types, pos)
        if len(evals_internal) != num_vibrational_modes:
            count_wrong_internal += 1
            diff = num_vibrational_modes - len(evals_internal)
            print(
                f"Internal {i}: vibrational modes: {diff}",
                "❌",
            )
            
        evals_geometric, eigvecs_geometric = get_modes_geometric(hessian, pos, atom_types)
        if len(evals_geometric) != num_vibrational_modes:
            count_wrong_geometric += 1
            diff = num_vibrational_modes - len(evals_geometric)
            print(
                f"Geometric {i}: vibrational modes: {diff}",
                "❌",
            )
        
        freq_vib_ase, modes_vib_ase, freq_all_ase, n_tr_ase = get_vibrational_modes_ase(hessian, pos, atom_types)
        if len(freq_vib_ase) != num_vibrational_modes:
            count_wrong_ase += 1
            diff = num_vibrational_modes - len(freq_vib_ase)
            print(
                f"ASE {i}: vibrational modes: {diff}",
                "❌",
            )
        

    print(f"Count wrong eckart svd: {count_wrong_eckart_svd}")
    print(f"Count wrong eckart qr: {count_wrong_eckart_qr}")
    print(f"Count wrong projection: {count_wrong_projection}")
    print(f"Count wrong internal: {count_wrong_internal}")
    print(f"Count wrong geometric: {count_wrong_geometric}")
    print(f"Count wrong ASE: {count_wrong_ase}")