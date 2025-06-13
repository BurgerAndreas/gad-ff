from mace.calculators import MACECalculator
from ase import build

mace_calc = MACECalculator(model_paths="mace_agnesi_small.model", enable_cueq=True, default_dtype="float64",device='cuda')
atoms = build.molecule('H2O')
atoms.calc = mace_calc
hessian = calc.get_hessian(atoms=atoms)