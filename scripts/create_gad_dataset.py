"""
Augmented Transition1x dataset (https://www.nature.com/articles/s41467-024-52481-5) -> GAD dataset.

For each datapoint (configuration) in the augmented Transition1x dataset:
- Compute the Hessian using the augmented-T1x-finetuned NewtonNet model (https://www.nature.com/articles/s41467-024-52481-5)
- Compute smallest eigenvalue and corresponding eigenvector of the Hessian
- Compute the GAD vector field
- Save datapoint with GAD vector field
"""

# Mace expects: energies in eV, forces in eV/Å, distances in Å
# RGD1 uses: Hartree, Angstrom

# 1 Hartree energy = 27.2114079527 eV
Hartree2eV = 27.2114079527
