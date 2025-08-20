#!/usr/bin/env python3
"""
Standalone script to compare Hessian-assisted optimization strategies using SciPy's BFGS.

This script compares:
0) No Hessian: Standard BFGS
1) Initial-only: Use predicted H_0 once, then pure BFGS updates
2) Periodic replace: Every k steps, reset H ← H_pred(x_k) with k ∈ {1, 3}

Metrics tracked:
- Gradient evaluations to converge
- Wall time (CPU/GPU)
- Success rate (%)
- Final gradient norm
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv, norm
import warnings
from dataclasses import dataclass
from typing import List, Tuple
from ase import Atoms

# Import the Equiformer calculator
from gadff.equiformer_ase_calculator import EquiformerASECalculator


@dataclass
class OptimizationMetrics:
    """Store optimization results and metrics."""

    success: bool
    n_grad_evals: int
    n_hess_evals: int
    wall_time: float
    final_energy: float
    final_gradient_norm: float
    convergence_history: List[Tuple[float, float]]  # (energy, grad_norm) pairs


class HessianBFGSWrapper:
    """Wrapper for SciPy BFGS with Hessian refresh capabilities."""

    def __init__(self, calculator, atoms_template, refresh_policy=None):
        self.calculator = calculator
        self.atoms_template = atoms_template
        self.refresh_policy = refresh_policy  # None, 'initial', or integer k
        self.reset_stats()

    def reset_stats(self):
        self.n_grad_evals = 0
        self.n_hess_evals = 0
        self.convergence_history = []
        self.iteration = 0
        self.start_time = None

    def objective_and_gradient(self, x):
        """Compute energy and forces, track gradient evaluations."""
        self.n_grad_evals += 1

        # Update atomic positions
        atoms = self.atoms_template.copy()
        atoms.positions = x.reshape(-1, 3)
        atoms.calc = self.calculator

        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            gradient = -forces.flatten()  # Negative of forces

            self.convergence_history.append((energy, norm(gradient)))
            return energy, gradient

        except Exception as e:
            print(f"Error in energy/force calculation: {e}")
            return np.inf, np.zeros_like(x)

    def get_hessian(self, x):
        """Get Hessian matrix from calculator."""
        self.n_hess_evals += 1
        try:
            atoms = self.atoms_template.copy()
            atoms.positions = x.reshape(-1, 3)
            atoms.calc = self.calculator
            hessian = self.calculator.get_hessian_autodiff(atoms)
            return hessian
        except Exception as e:
            print(f"Error in Hessian calculation: {e}")
            return np.eye(len(x))

    def callback(self, x):
        """Callback function for SciPy minimize."""
        self.iteration += 1
        # For periodic refresh strategies
        if (
            isinstance(self.refresh_policy, int)
            and self.iteration % self.refresh_policy == 0
        ):
            # This is handled in the optimize method through hessp
            pass

    def hessp_initial_only(self, x, p):
        """Hessian-vector product for initial-only strategy."""
        if self.iteration == 0:
            H = self.get_hessian(x)
            return H @ p
        else:
            # Return None to let BFGS handle it
            return None

    def optimize(self):
        """Run optimization using SciPy's BFGS."""
        self.start_time = time.time()
        x0 = self.atoms_template.positions.flatten()

        # Choose optimization strategy
        if self.refresh_policy is None:
            # Standard BFGS
            result = minimize(
                fun=self.objective_and_gradient,
                x0=x0,
                method="BFGS",
                jac=True,
                callback=self.callback,
                options={"gtol": 1e-3, "maxiter": 100},
            )

        elif self.refresh_policy == "initial":
            # Initial Hessian only
            try:
                H0_inv = inv(self.get_hessian(x0))
                result = minimize(
                    fun=self.objective_and_gradient,
                    x0=x0,
                    method="BFGS",
                    jac=True,
                    callback=self.callback,
                    options={"gtol": 1e-3, "maxiter": 100, "hess_inv0": H0_inv},
                )
            except Exception:
                # Fallback to standard BFGS if Hessian fails
                result = minimize(
                    fun=self.objective_and_gradient,
                    x0=x0,
                    method="BFGS",
                    jac=True,
                    callback=self.callback,
                    options={"gtol": 1e-3, "maxiter": 100},
                )

        elif isinstance(self.refresh_policy, int):
            # Periodic refresh - use multiple restarts
            best_result = None
            best_energy = np.inf
            x_current = x0.copy()
            total_nfev = 0
            last_result = None

            for restart in range(10):  # Max 10 restarts
                try:
                    # Get current Hessian
                    H_inv = inv(self.get_hessian(x_current))

                    # Run BFGS for k steps
                    sub_result = minimize(
                        fun=self.objective_and_gradient,
                        x0=x_current,
                        method="BFGS",
                        jac=True,
                        options={
                            "gtol": 1e-3,
                            "maxiter": self.refresh_policy,
                            "hess_inv0": H_inv,
                        },
                    )

                    last_result = sub_result
                    total_nfev += sub_result.nfev
                    x_current = sub_result.x

                    if sub_result.fun < best_energy:
                        best_energy = sub_result.fun
                        best_result = sub_result

                    # Check convergence
                    if norm(sub_result.jac) < 1e-3:
                        break

                except Exception:
                    break

            # Use best result if available, otherwise last result, otherwise create a dummy result
            if best_result is not None:
                result = best_result
                result.nfev = total_nfev
            elif last_result is not None:
                result = last_result
                result.nfev = total_nfev
            else:
                # Create a dummy failed result
                from scipy.optimize import OptimizeResult

                result = OptimizeResult()
                result.success = False
                result.fun = np.inf
                result.x = x0
                result.jac = np.zeros_like(x0)
                result.nfev = total_nfev

        else:
            raise ValueError(f"Unknown refresh policy: {self.refresh_policy}")

        wall_time = time.time() - self.start_time

        return OptimizationMetrics(
            success=result.success,
            n_grad_evals=result.nfev,
            n_hess_evals=self.n_hess_evals,
            wall_time=wall_time,
            final_energy=result.fun,
            final_gradient_norm=norm(result.jac),
            convergence_history=self.convergence_history,
        )


def create_test_molecules():
    """Create a set of test molecules for optimization (more perturbed from equilibrium)."""
    molecules = {}

    # Water molecule (more significantly perturbed from equilibrium)
    molecules["H2O"] = Atoms(
        symbols="H2O",
        positions=[
            [0.0, 0.0, 0.0],  # O
            [0.5, 1.2, 0.8],  # H (stretched bonds)
            [-0.5, -1.2, 0.8],  # H (stretched bonds)
        ],
    )

    # Ammonia molecule (compressed)
    molecules["NH3"] = Atoms(
        symbols="NH3",
        positions=[
            [0.0, 0.0, 0.0],  # N
            [0.0, 0.6, -0.2],  # H (compressed)
            [-0.5, -0.3, -0.2],  # H
            [0.5, -0.3, -0.2],  # H
        ],
    )

    # Methane molecule (distorted)
    molecules["CH4"] = Atoms(
        symbols="CH4",
        positions=[
            [0.0, 0.0, 0.0],  # C
            [0.8, 0.8, 0.8],  # H (stretched)
            [-0.4, -0.4, 0.8],  # H (compressed)
            [-0.8, 0.8, -0.4],  # H
            [0.4, -0.8, -0.8],  # H
        ],
    )

    return molecules


def run_comparison_study():
    """Run the complete comparison study."""
    print("=" * 60)
    print("Hessian-Assisted Optimization Comparison Study")
    print("=" * 60)

    # Initialize calculator
    calculator = EquiformerASECalculator()

    # Define optimization strategies
    strategies = {
        "No Hessian": None,
        "Initial Only": "initial",
        "Refresh k=1": 1,
        "Refresh k=3": 3,
    }

    # Get test molecules
    molecules = create_test_molecules()

    # Store all results
    all_results = {}

    for mol_name, atoms in molecules.items():
        print(f"\n--- Optimizing {mol_name} ---")
        all_results[mol_name] = {}

        for strategy_name, refresh_policy in strategies.items():
            print(f"  Strategy: {strategy_name}")

            # Create optimizer wrapper
            optimizer = HessianBFGSWrapper(
                calculator=calculator,
                atoms_template=atoms.copy(),
                refresh_policy=refresh_policy,
            )

            # Run optimization
            try:
                result = optimizer.optimize()
                all_results[mol_name][strategy_name] = result

                # Print summary
                print(f"    Success: {result.success}")
                print(f"    Gradient evals: {result.n_grad_evals}")
                print(f"    Hessian evals: {result.n_hess_evals}")
                print(f"    Wall time: {result.wall_time:.3f}s")
                print(f"    Final |grad|: {result.final_gradient_norm:.2e}")

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results[mol_name][strategy_name] = None

    return all_results


def analyze_results(all_results):
    """Analyze and print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    strategies = ["No Hessian", "Initial Only", "Refresh k=1", "Refresh k=3"]

    # Collect metrics
    metrics = {
        strategy: {
            "success_rate": [],
            "grad_evals": [],
            "hess_evals": [],
            "wall_time": [],
        }
        for strategy in strategies
    }

    for mol_name, mol_results in all_results.items():
        for strategy in strategies:
            result = mol_results.get(strategy)
            if result is not None:
                metrics[strategy]["success_rate"].append(1 if result.success else 0)
                if result.success:
                    metrics[strategy]["grad_evals"].append(result.n_grad_evals)
                    metrics[strategy]["hess_evals"].append(result.n_hess_evals)
                    metrics[strategy]["wall_time"].append(result.wall_time)

    # Print summary table
    print(
        f"{'Strategy':<15} {'Success%':<10} {'Grad Evals':<12} {'Hess Evals':<12} {'Wall Time':<12}"
    )
    print("-" * 75)

    for strategy in strategies:
        m = metrics[strategy]

        success_rate = np.mean(m["success_rate"]) * 100 if m["success_rate"] else 0
        avg_grad_evals = np.mean(m["grad_evals"]) if m["grad_evals"] else 0
        avg_hess_evals = np.mean(m["hess_evals"]) if m["hess_evals"] else 0
        avg_wall_time = np.mean(m["wall_time"]) if m["wall_time"] else 0

        print(
            f"{strategy:<15} {success_rate:>7.1f}% {avg_grad_evals:>10.1f} {avg_hess_evals:>10.1f} {avg_wall_time:>9.3f}s"
        )

    # Detailed analysis for successful runs
    print(
        f"\n{'Strategy':<15} {'Min Grad':<12} {'Max Grad':<12} {'Std Grad':<12} {'Efficiency':<12}"
    )
    print("-" * 75)

    for strategy in strategies:
        grad_evals = metrics[strategy]["grad_evals"]
        hess_evals = metrics[strategy]["hess_evals"]
        if grad_evals:
            min_grad = min(grad_evals)
            max_grad = max(grad_evals)
            std_grad = np.std(grad_evals)
            # Efficiency metric: lower is better (grad_evals + 10*hess_evals as rough cost)
            efficiency = np.mean([g + 10 * h for g, h in zip(grad_evals, hess_evals)])
            print(
                f"{strategy:<15} {min_grad:>10.1f} {max_grad:>10.1f} {std_grad:>10.1f} {efficiency:>10.1f}"
            )


def main():
    """Main execution function."""
    # Check if calculator can be loaded
    try:
        print("Initializing Equiformer calculator...")
        EquiformerASECalculator()
        print("✓ Calculator loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load calculator: {e}")
        print("Please ensure the Equiformer model checkpoint is available.")
        return

    # Run comparison study
    results = run_comparison_study()

    # Analyze results
    analyze_results(results)

    print(f"\n{'=' * 60}")
    print("Study completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
