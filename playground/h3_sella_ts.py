#!/usr/bin/env python3
"""
H₃ Transition State Calculation using Sella
Reaction: H + H₂ → H₂ + H

This script finds and verifies the transition state for hydrogen atom exchange.
"""

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations
from sella import Sella
import os

thisfolder = os.path.dirname(os.path.abspath(__file__))
plotfolder = os.path.join(thisfolder, "ts_plots")
logfolder = os.path.join(thisfolder, "ts_logs")


def setup_h3_system():
    """Set up H₃ system from existing structure or create initial guess."""

    # Try to load existing structure, fallback to manual creation
    try:
        atoms = read("h3_transition_state.xyz")
        print("Loaded existing H₃ structure from h3_transition_state.xyz")
    except FileNotFoundError:
        print("Creating initial H₃ structure manually")
        # Create linear H₃ structure as initial guess for TS
        positions = np.array(
            [
                [-0.5, 0.0, 0.0],  # H1
                [0.0, 0.0, 0.0],  # H2 (central)
                [0.5, 0.0, 0.0],  # H3
            ]
        )
        atoms = Atoms("H3", positions=positions)
        atoms.center(vacuum=5.0)  # Add vacuum around the molecule

    # Set up EMT calculator (fast and reasonable for H₃)
    atoms.calc = EMT()

    return atoms


def run_sella_optimization(atoms):
    """Run Sella optimization to find transition state."""

    print("\n" + "=" * 60)
    print("STARTING SELLA TRANSITION STATE SEARCH")
    print("=" * 60)

    initial_energy = atoms.get_potential_energy()
    print(f"Initial energy: {initial_energy:.6f} eV")

    # Set up Sella optimizer for first-order saddle point
    dyn = Sella(
        atoms,
        order=1,  # First-order saddle point (transition state)
        # trajectory=os.path.join(logfolder, 'h3_sella_ts.traj'),
        # logfile=os.path.join(logfolder, 'h3_sella_ts.log'),
        eta=1e-4,  # Finite difference step size
        delta0=1e-2,  # Initial trust radius
        gamma=1e-3,  # Convergence tolerance for iterative diagonalization
        rho_inc=1.1,  # Trust radius increase factor
        rho_dec=2.0,  # Trust radius decrease factor
        sigma_inc=1.5,  # Trust radius increase multiplier
        sigma_dec=0.5,  # Trust radius decrease multiplier
    )

    # Run optimization with tight convergence
    print("Running Sella optimization")
    dyn.run(fmax=1e-4, steps=1000)  # Tight force convergence

    final_energy = atoms.get_potential_energy()
    max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))

    print(f"Final energy: {final_energy:.6f} eV")
    print(f"Maximum force: {max_force:.6f} eV/Å")
    print("Sella optimization completed!")

    return atoms


def verify_transition_state(atoms):
    """Verify the structure is a transition state via frequency analysis."""

    print("\n" + "=" * 60)
    print("VERIFYING TRANSITION STATE")
    print("=" * 60)

    print("Performing frequency analysis")

    # Run vibrational analysis on all atoms
    vib = Vibrations(
        atoms,
        #  name=os.path.join(logfolder, 'h3_ts_verification')
    )
    vib.run()

    # Get frequencies and energies
    vib_data = vib.get_vibrations()
    frequencies = vib_data.get_frequencies()  # in cm⁻¹
    energies = vib_data.get_energies()  # in eV

    print("\nVibrational Analysis Results:")
    print("-" * 50)
    print("Mode    Frequency (cm⁻¹)    Energy (meV)")
    print("-" * 50)

    negative_freq_count = 0
    imaginary_frequencies = []

    for i, (freq, energy) in enumerate(zip(frequencies, energies)):
        energy_mev = energy * 1000  # Convert to meV
        if energy < 0:  # Negative eigenvalue = imaginary frequency
            negative_freq_count += 1
            imaginary_frequencies.append(freq)
            print(f"{i:3d}    {freq:8.1f}i        {abs(energy_mev):8.1f}")
        else:
            print(f"{i:3d}    {freq:8.1f}         {energy_mev:8.1f}")

    print("-" * 50)

    # Evaluate transition state criteria
    print(f"\nTransition State Verification:")
    print(f"Number of imaginary frequencies: {negative_freq_count}")

    if negative_freq_count == 1:
        print("✓ SUCCESS: Structure has exactly 1 imaginary frequency")
        print(
            "✓ This confirms the structure is a first-order saddle point (transition state)"
        )
        print(f"✓ Imaginary frequency: {imaginary_frequencies[0]:.1f}i cm⁻¹")

        # Write the imaginary mode for visualization
        vib.write_mode(0)
        print("✓ Transition state mode written to 'h3_ts_verification.0.traj'")
        print("  View with: ase gui h3_ts_verification.0.traj")

        status = "VERIFIED TRANSITION STATE"

    elif negative_freq_count == 0:
        print("✗ FAILURE: Structure has no imaginary frequencies")
        print("✗ This appears to be a minimum, not a transition state")
        status = "MINIMUM (NOT TRANSITION STATE)"

    else:
        print(f"✗ FAILURE: Structure has {negative_freq_count} imaginary frequencies")
        print("✗ This appears to be a higher-order saddle point")
        print(
            f"✗ Imaginary frequencies: {[f'{freq:.1f}i' for freq in imaginary_frequencies]} cm⁻¹"
        )
        status = f"HIGHER-ORDER SADDLE POINT ({negative_freq_count} imaginary modes)"

    # Save detailed summary
    vib.summary(log="h3_ts_verification_summary.txt")
    print("Detailed vibrational summary saved to 'h3_ts_verification_summary.txt'")

    # Clean up temporary files
    vib.clean()

    return status, negative_freq_count, imaginary_frequencies


def main():
    """Main function to run H₃ transition state calculation."""

    print("H₃ Transition State Calculation using Sella")
    print("Reaction: H + H₂ → H₂ + H")
    print("=" * 60)

    # 1. Set up H₃ system
    atoms = setup_h3_system()
    print(f"H₃ system set up with {len(atoms)} atoms")

    # Save initial structure
    # write(os.path.join(logfolder, 'h3_initial_guess.xyz'), atoms)
    # print("Initial structure saved to 'h3_initial_guess.xyz'")

    # 2. Run Sella optimization
    atoms = run_sella_optimization(atoms)

    # Save optimized structure
    # write('h3_optimized_ts.xyz', atoms)
    # print("Optimized structure saved to 'h3_optimized_ts.xyz'")

    # 3. Verify transition state
    status, n_imag, imag_freqs = verify_transition_state(atoms)

    # 4. Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
    print(f"Max force: {np.max(np.linalg.norm(atoms.get_forces(), axis=1)):.6f} eV/Å")
    print(f"Status: {status}")

    if n_imag == 1:
        print(f"Transition state frequency: {imag_freqs[0]:.1f}i cm⁻¹")
        print("✓ H₃ transition state calculation SUCCESSFUL!")
    else:
        print("✗ H₃ transition state calculation needs refinement")
        print("Suggestions:")
        print("  - Try different initial geometry")
        print("  - Adjust Sella optimization parameters")
        print("  - Consider using a higher-level calculator")

    print("=" * 60)


if __name__ == "__main__":
    main()
