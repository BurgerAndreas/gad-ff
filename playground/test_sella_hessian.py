from sella import Sella

# Assuming you have an ML calculator with Hessian capability
ml_calc = EquiformerASECalculator(checkpoint_path="model.pt")
atoms.calc = ml_calc


# Create wrapper function
def ml_hessian_function(atoms):
    return ml_calc.get_hessian_autodiff(atoms)


# Use with Sella
dyn = Sella(
    atoms,
    hessian_function=ml_hessian_function,
    order=1,  # For saddle point optimization
    eig=True,  # Enable eigenvalue computation
)
