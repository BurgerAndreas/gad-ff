
"""
Augmented Transition1x dataset (https://www.nature.com/articles/s41467-024-52481-5) -> GAD dataset.

For each datapoint (configuration) in the augmented Transition1x dataset:
- Compute the Hessian using the augmented-T1x-finetuned NewtonNet model (https://www.nature.com/articles/s41467-024-52481-5)
- Compute smallest eigenvalue and corresponding eigenvector of the Hessian
- Compute the GAD vector field
- Save datapoint with GAD vector field
"""