"""
Zero-point energy (ZPE) for a molecular geometry is computed from the vibrational frequencies of the molecule at that geometry. 

The steps are:
a. Optimize the molecular geometry (find the equilibrium structure).
b. Compute the vibrational frequencies by evaluating the Hessian (second derivative of energy with respect to nuclear displacements).
c. The zero-point energy is the sum over all vibrational mode with frequency v_i and energy (1/2) h v_i
ZPE = (1/2) Σ h v_i


In practice, we:
1. Start from T1x val reactants.
2. Relax the geometry further using BFGS with EquiformerV2
(if necessary we will have to use DFT via PySCF /ssd/Code/ReactBench/dependencies/pysisyphus/pysisyphus/calculators/PySCF.py )
3. Confirm we converged to max_force_dft < 10^-2 eV/Angstrom, ideally 10^-3 eV/Angstrom
4. Save the relaxed geometry
5. Compute the Hessian using DFT via PySCF, save the Hessian
6. Compute the Hessian using various models (see scripts/eval_horm.py)
7. Mass-weight and Eckart-project the Hessians for each method
8. Compute the ZPE for each method
9. Save ZPEs in dataframe and to csv
"""

import argparse