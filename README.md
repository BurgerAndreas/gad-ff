# GAD-FF

Directly predict the lowest Hessian eigenvalues and their eigenvectors for fast transition state search.
Based on the HORM EquiformerV2 model, finetuned with extra prediction heads on the HORM DFT Hessians.

## Installation

Setup the environment [docs/cluster.md](docs/cluster.md)


## Available checkpoints

- `ckpt/eqv2.ckpt`: HORM EquiformerV2 finetuned on the HORM Hessian dataset. Not trained to predict the Hessian eigenvalues and eigenvectors! Will give random results. Can be used with autograd Hessian.


## Use our model

Use our model, that 
```bash
# download from HuggingFace
https://huggingface.co/andreasburger/heigen
```

See [example_inference.py](example_inference.py) for a full example how to use our model.

```python
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
from torch_geometric.loader import DataLoader as TGDataLoader

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props

config_path = "configs/equiformer_v2.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
model_config = config["model"]
model = EquiformerV2_OC20(**model_config)

checkpoint_path = "ckpt/eqv2.ckpt"
model.load_state_dict(
    torch.load(checkpoint_path, weights_only=True)["state_dict"], 
    strict=False
)
model.eval()
model.to("cuda")

# random batch of data
n_atoms = 10
data = TGData(
    pos=torch.randn(n_atoms, 3),
    batch=torch.zeros(n_atoms),
    one_hot=torch.randint(0, 4, (n_atoms, 4)),
    natoms=n_atoms,
    # just needs be a placeholder that decides the output energy shape
    energy=torch.randn(1), 
)
data = Batch.from_data_list([data])
# dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)
# data = next(iter(dataloader))

data = data.to(model.device)
energy, forces, eigenpred = model.forward(data, eigen=True)

# Compute GAD
v = eigenpred["eigvec_1"].reshape(B, -1)
forces = forces.reshape(B, -1)
# −∇V(x) + 2(∇V, v(x))v(x)
gad = -forces + 2 * torch.einsum("bi,bi->b", forces, v) * v

```


## Recreate our models

Code: https://github.com/BurgerAndreas/gad-ff

Recreate our Hessian-eigen dataset [docs/eigen_dataset.md](docs/eigen_dataset.md)

Recreate our direct-prediction-Hessian-eigen model [docs/training.md](docs/training.md)


## Citation

```bibtex
TODO
```

```bibtex
HORM
```