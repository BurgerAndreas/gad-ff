from ase import build
import torch
import numpy as np
import os
from mace.calculators import MACECalculator


home = os.path.expanduser("~")
model_path = os.path.join(home, ".cache/mace/MACE-OFF23_medium.model")

# Interface if you already have a model downloaded 
# or just want to wrap the calculator around it
calc = MACECalculator(
    model_paths=model_path, 
    # enable_cueq=True, 
    default_dtype="float64", device='cuda',
)
atoms = build.molecule('H2O')
atoms.calc = calc
hessian = calc.get_hessian(atoms=atoms)
print(hessian.shape)

model = calc.models[0]
print("Model name:", model.__class__.__name__)