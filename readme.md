# GAD-FF

## Installation

Setup the environment [documentation/cluster.md](documentation/cluster.md)





## Use our model

Use our model, that directly predicts the Hessian eigenvalues and eigenvectors
(based on the HORM EquiformerV2 model)
```bash
# download checkpoint
TODO

# run inference
TODO
```

## Recreate our models

Recreate our Hessian-eigen dataset [documentation/eigen_dataset.md](documentation/eigen_dataset.md)

Recreate our direct-prediction-Hessian-eigen model [documentation/training.md](documentation/training.md)

## FAQ

- Composition split vs. Conformation split. The T1x dataset for training is split in two different ways. 
    - Molecular compositions: Harder. Tests generalization to unseen configurations. Reactant, product, and transition state geometries are all in the same set.
    - Molecular conformations: Easier. Tests generalization to unseen parts of the potential energy surface. Reactant and transition state might be train, and product might be test.

## Citation

```bibtex
TODO
```

```bibtex
HORM
```