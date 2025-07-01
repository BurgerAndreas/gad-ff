from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations
from ase.optimize import BFGS
import numpy as np
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
import os

thisfolder = os.path.dirname(os.path.abspath(__file__))
plotfolder = os.path.join(thisfolder, "ts_plots")
logfolder = os.path.join(thisfolder, "ts_logs")

from sella import Sella, Constraints


def run_vibrational_analysis(slab, adsorbate_indices):
    print(f"Analyzing vibrations for adsorbate atom: {adsorbate_indices}")

    vib = Vibrations(
        slab,
        indices=adsorbate_indices,
        # name=os.path.join(logfolder, 'ts_verification')
    )
    vib.run()

    # Get vibrational data
    vib_data = vib.get_vibrations()
    frequencies = vib_data.get_frequencies()  # in cm^-1
    energies = vib_data.get_energies()  # in eV
    vib.summary()

    # print("\nVibrational Analysis Results:")
    # print("-" * 40)
    # print("Mode    Frequency (cm⁻¹)    Energy (eV)")
    # print("-" * 40)

    # negative_freq_count = 0
    # imaginary_frequencies = []

    # for i, (freq, energy) in enumerate(zip(frequencies, energies)):
    #     # Check if frequency is complex (imaginary) or if energy has imaginary component
    #     if np.iscomplexobj(freq) and np.imag(freq) != 0:
    #         negative_freq_count += 1
    #         imaginary_frequencies.append(np.imag(freq))  # Store the imaginary part
    #         print(f"{i:3d}    {np.imag(freq):8.1f}i        {abs(np.imag(energy)):8.4f}")
    #     elif np.iscomplexobj(energy) and np.imag(energy) != 0:
    #         negative_freq_count += 1
    #         imaginary_frequencies.append(np.imag(freq) if np.iscomplexobj(freq) else freq)
    #         print(f"{i:3d}    {freq:8.1f}i        {abs(np.imag(energy)):8.4f}")
    #     elif (not np.iscomplexobj(energy) and energy < 0) or (
    #         not np.iscomplexobj(freq) and freq < 0
    #     ):
    #         negative_freq_count += 1
    #         imaginary_frequencies.append(abs(freq))
    #         print(f"{i:3d}    {abs(freq):8.1f}i        {abs(energy):8.4f}")
    #     else:
    #         # Real positive frequency
    #         real_freq = np.real(freq) if np.iscomplexobj(freq) else freq
    #         real_energy = np.real(energy) if np.iscomplexobj(energy) else energy
    #         print(f"{i:3d}    {real_freq:8.1f}         {real_energy:8.4f}")

    # print("-" * 40)

    # # Verify transition state criteria
    # print(f"\nTransition State Verification:")
    # print(f"Number of imaginary frequencies: {negative_freq_count}")

    # if negative_freq_count == 1:
    #     print("✓ SUCCESS: Structure has exactly 1 imaginary frequency")
    #     print(
    #         "✓ This confirms the structure is a first-order saddle point (transition state)"
    #     )
    #     print(f"✓ Imaginary frequency: {imaginary_frequencies[0]:.1f}i cm⁻¹")
    # elif negative_freq_count == 0:
    #     print("✗ FAILURE: Structure has no imaginary frequencies")
    #     print("✗ This appears to be a minimum, not a transition state")
    #     print(
    #         "  Try: larger initial displacement, different starting geometry, or adjust Sella parameters"
    #     )
    # elif negative_freq_count > 1:
    #     print(f"✗ FAILURE: Structure has {negative_freq_count} imaginary frequencies")
    #     print("✗ This appears to be a higher-order saddle point")
    #     print(
    #         f"✗ Imaginary frequencies: {[f'{freq:.1f}i' for freq in imaginary_frequencies]} cm⁻¹"
    #     )

    # # Summary
    # print(f"\nSummary:")
    # print(f"Initial energy (minimum): {min_energy:.6f} eV")  # From previous calculation
    # print(f"Final energy (TS search): {slab.get_potential_energy():.6f} eV")
    # print(f"Maximum force: {np.max(np.linalg.norm(slab.get_forces(), axis=1)):.6f} eV/Å")

    # if negative_freq_count == 1:
    #     print("STATUS: ✓ VERIFIED TRANSITION STATE")

    #     # If we found a transition state, let's visualize the imaginary mode
    #     print("\nWriting transition state mode visualization...")
    #     vib.write_mode(0)  # Write the imaginary mode to the traj
    # else:
    #     print("STATUS: ✗ NOT A TRANSITION STATE")

    # print("=" * 60)

    # Optionally save the vibrational summary
    # vib.summary(log=os.path.join(logfolder, "cu_ts_verification_summary.txt"))
    # print("Detailed vibrational summary saved to 'ts_logs/cu_ts_verification_summary.txt'")

    # Clean up temporary vibration files
    vib.clean()
    
def plot_structure(atoms, filename, title_prefix="Structure"):
    """Plot atomic structure with side and top views."""
    print(f"\nPlotting {title_prefix.lower()}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Set up colors and radii - adsorbate (last atom) is red and smaller
    colors = ["orange"] * len(atoms)  # All atoms orange (Cu color)
    colors[-1] = "red"  # Adsorbate is red

    radii = [1.0] * len(atoms)  # All atoms normal size
    radii[-1] = 0.8  # Adsorbate is smaller

    # Side view
    plot_atoms(atoms, ax1, colors=colors, radii=radii, rotation=("0x,0y,0z"))
    ax1.set_title(f"{title_prefix} - Side View")
    ax1.set_xlabel("x (Å)")
    ax1.set_ylabel("z (Å)")

    # Top view
    plot_atoms(atoms, ax2, colors=colors, radii=radii, rotation=("90x,0y,0z"))
    ax2.set_title(f"{title_prefix} - Top View")
    ax2.set_xlabel("x (Å)")
    ax2.set_ylabel("y (Å)")

    plt.tight_layout()
    filepath = os.path.join(plotfolder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"{title_prefix} saved as '{filepath}'")

def find_ts_with_sella(slab, cons):
    ts_slab = slab.copy()
    # Set up a Sella Dynamics object with improved parameters for TS search
    print("\nStarting Sella optimization to find transition state...")
    dyn = Sella(
        ts_slab,
        constraints=cons,
        # trajectory=os.path.join(logfolder, "cu_sella_ts.traj"),
        order=1,  # Explicitly search for first-order saddle point
        eta=5e-5,  # Smaller finite difference step for higher accuracy
        delta0=5e-3,  # Larger initial trust radius for TS search
        gamma=0.1,  # Much tighter convergence for iterative diagonalization
        rho_inc=1.05,  # More conservative trust radius adjustment
        rho_dec=3.0,  # Allow larger trust radius changes
        sigma_inc=1.3,  # Larger trust radius increases
        sigma_dec=0.5,  # More aggressive trust radius decreases
    )

    # Run with much tighter convergence
    dyn.run(1e-5, 2000)  # Much stricter convergence criterion
    print("Sella optimization completed!")
    return ts_slab

if __name__ == "__main__":
    print("Setting up system for transition state search...")

    # Set up your system as an ASE atoms object
    slab = fcc111("Cu", (4, 4, 2), vacuum=7.5)  # Smaller slab for efficiency

    # Set up your calculator
    slab.calc = EMT()

    # First, let's find the minimum energy structure
    print("Finding minimum energy structure first...")
    cons_min = Constraints(slab)
    for atom in slab:
        cons_min.fix_translation(atom.index)
    add_adsorbate(slab, "Cu", 2.0, "bridge")

    # Optimize to minimum first
    opt_min = BFGS(
        slab,
        # logfile=os.path.join(logfolder, 'cu_minimize.log'),
        # trajectory=os.path.join(logfolder, 'cu_minimize.traj')
    )
    opt_min.run(fmax=0.01)

    min_energy = slab.get_potential_energy()
    print(f"Minimum energy: {min_energy:.6f} eV")

    # Now create a perturbed structure to search for transition state
    # Move the adsorbate slightly to create an initial guess for TS
    adsorbate_index = len(slab) - 1  # Last atom is the adsorbate
    original_pos = slab.positions[adsorbate_index].copy()

    # Perturb the adsorbate position more significantly to break symmetry
    print("Creating initial guess for transition state...")
    displacement = np.array([0.8, 0.4, 0.3])  # Larger displacement to escape minimum basin
    slab.positions[adsorbate_index] += displacement

    print(f"Adsorbate moved from {original_pos} to {slab.positions[adsorbate_index]}")

    # Set up constraints for transition state search
    cons = Constraints(slab)
    for atom in slab:
        if atom.position[2] < slab.cell[2, 2] / 2.0:  # Fix slab atoms
            cons.fix_translation(atom.index)

    ts_slab = find_ts_with_sella(slab, cons)

    plot_structure(ts_slab, "cu_sella_ts.png", "Sella TS")

    # Verify that the final structure is a transition state
    print("\n# VERIFYING TRANSITION STATE")

    adsorbate_indices = [adsorbate_index]
    run_vibrational_analysis(slab, adsorbate_indices)

    