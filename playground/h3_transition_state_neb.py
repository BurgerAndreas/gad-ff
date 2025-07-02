#!/usr/bin/env python3
"""
ASE NEB script to find transition state for H3 system: H + H2 -> H2 + H
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.mep import NEB  # Updated import
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.io import write, read
from ase.visualize import view
import os

thisfolder = os.path.dirname(os.path.abspath(__file__))
plotfolder = os.path.join(thisfolder, "ts_plots")
logfolder = os.path.join(thisfolder, "ts_logs")


def create_h3_endpoints():
    """Create initial and final states for H + H2 -> H2 + H reaction"""

    # Reactant state: H (far left) + H2 (bonded, right side)
    # H2 bond length ~0.74 Å
    reactant = Atoms(
        "HHH",
        positions=[
            [-2.0, 0.0, 0.0],  # Approaching H atom
            [1.0, 0.0, 0.0],  # H2 molecule
            [1.74, 0.0, 0.0],
        ],  # H2 molecule
        cell=[8, 6, 6],
        pbc=False,
    )

    # Product state: H2 (left side) + H (far right)
    product = Atoms(
        "HHH",
        positions=[
            [-1.74, 0.0, 0.0],  # H2 molecule
            [-1.0, 0.0, 0.0],  # H2 molecule
            [2.0, 0.0, 0.0],
        ],  # Departing H atom
        cell=[8, 6, 6],
        pbc=False,
    )

    return reactant, product


def setup_neb_calculation(reactant, product, nimages=7):
    """Set up NEB calculation with intermediate images"""

    # Create list of images
    images = [reactant.copy()]

    # Add intermediate images
    for i in range(nimages - 2):
        images.append(reactant.copy())

    # Add final image
    images.append(product.copy())

    # Create NEB object
    neb = NEB(images, parallel=False, climb=True)  # climb=True for better TS

    # Interpolate between initial and final states
    neb.interpolate()

    return neb, images


def run_neb_optimization(neb, images, fmax=0.05, steps=200):
    """Run NEB optimization to find minimum energy path"""

    # Set separate calculator for each image (required in newer ASE)
    for image in images:
        image.calc = EMT()  # Each image gets its own calculator instance

    # Set up optimizer
    optimizer = BFGS(neb)

    print(f"Starting NEB optimization")
    print(f"Target force convergence: {fmax} eV/Å")
    print(f"Maximum steps: {steps}")

    # Run optimization
    optimizer.run(fmax=fmax, steps=steps)

    print("NEB optimization completed!")
    return optimizer


def analyze_results(images):
    """Analyze NEB results and find transition state"""

    # Calculate energies along the path
    energies = []
    for i, image in enumerate(images):
        energy = image.get_potential_energy()
        energies.append(energy)
        print(f"Image {i}: Energy = {energy:.4f} eV")

    # Convert to relative energies (kcal/mol)
    energies = np.array(energies)
    rel_energies = (energies - energies[0]) * 23.06  # eV to kcal/mol

    # Find transition state (highest energy image)
    ts_index = np.argmax(rel_energies)
    ts_energy = rel_energies[ts_index]

    print(f"\nTransition State Analysis:")
    print(f"TS found at image {ts_index}")
    print(f"Activation barrier: {ts_energy:.2f} kcal/mol")
    print(f"Reaction energy: {rel_energies[-1]:.2f} kcal/mol")

    return rel_energies, ts_index


def plot_energy_profile(rel_energies):
    """Plot the reaction energy profile"""

    plt.figure(figsize=(10, 6))
    reaction_coord = np.arange(len(rel_energies))

    plt.plot(reaction_coord, rel_energies, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Reaction Coordinate (Image Number)")
    plt.ylabel("Relative Energy (kcal/mol)")
    plt.title("H₃ Reaction Energy Profile: H + H₂ → H₂ + H")
    plt.grid(True, alpha=0.3)

    # Mark transition state
    ts_idx = np.argmax(rel_energies)
    plt.plot(
        ts_idx,
        rel_energies[ts_idx],
        "ro",
        markersize=12,
        label=f"TS: {rel_energies[ts_idx]:.1f} kcal/mol",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(plotfolder, "h3_energy_profile.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    return plt.gcf()


def save_geometries(images):
    """Save all geometries along the reaction path"""

    # Save individual xyz files
    for i, image in enumerate(images):
        write(os.path.join(logfolder, f"h3_image_{i:02d}.xyz"), image)

    # Save trajectory file
    write(os.path.join(logfolder, "h3_reaction_path.xyz"), images)

    # Save transition state separately
    ts_index = np.argmax([img.get_potential_energy() for img in images])
    write(os.path.join(logfolder, "h3_transition_state.xyz"), images[ts_index])

    print(f"Geometries saved:")
    print(f"- Individual images: h3_image_XX.xyz")
    print(f"- Full trajectory: h3_reaction_path.xyz")
    print(f"- Transition state: h3_transition_state.xyz")


def main():
    """Main function to run the complete NEB calculation"""

    print("=" * 60)
    print("ASE NEB Calculation for H₃ Transition State")
    print("Reaction: H + H₂ → H₂ + H")
    print("=" * 60)

    # Step 1: Create endpoint geometries
    print("\n1. Creating reactant and product geometries")
    reactant, product = create_h3_endpoints()

    # Save endpoints
    # write(os.path.join(logfolder, 'h3_reactant.xyz'), reactant)
    # write(os.path.join(logfolder, 'h3_product.xyz'), product)
    # print("   Saved: h3_reactant.xyz, h3_product.xyz")

    # Step 2: Set up NEB calculation
    print("\n2. Setting up NEB calculation")
    nimages = 7
    neb, images = setup_neb_calculation(reactant, product, nimages)
    print(f"   Created NEB chain with {nimages} images")

    # Step 3: Run optimization
    print("\n3. Running NEB optimization")
    optimizer = run_neb_optimization(neb, images, fmax=0.05, steps=200)

    # Step 4: Analyze results
    print("\n4. Analyzing results")
    rel_energies, ts_index = analyze_results(images)

    # Step 5: Create plots
    print("\n5. Creating energy profile plot")
    plot_energy_profile(rel_energies)

    # Step 6: Save geometries
    # print("\n6. Saving geometries")
    # save_geometries(images)

    print("\n" + "=" * 60)
    print("NEB calculation completed successfully!")
    print("Check the generated files:")
    print("- h3_energy_profile.png (reaction profile)")
    print("- h3_transition_state.xyz (TS geometry)")
    print("=" * 60)


if __name__ == "__main__":
    main()
