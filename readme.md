# GAD-FF

## Installation
See [documentation/cluster.md](documentation/cluster.md)

## Run

```bash
python scripts/equiformer_eigen_dataset.py
```

## FAQ

- Composition split vs. Conformation split. The T1x dataset for training is split in two different ways. 
    - Molecular compositions: Harder. Tests generalization to unseen configurations. Reactant, product, and transition state geometries are all in the same set.
    - Molecular conformations: Easier. Tests generalization to unseen parts of the potential energy surface. Reactant and transition state might be train, and product might be test.
