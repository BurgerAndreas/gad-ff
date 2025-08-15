# GAD-FF

Directly predict the lowest Hessian eigenvalues and their eigenvectors for fast transition state search.
Based on the HORM EquiformerV2 model, finetuned with extra prediction heads on the HORM DFT Hessians.

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
uv venv hesspred --python 3.11
source hesspred/bin/activate
uv pip install --upgrade pip

uv pip install torch==2.4.1 torch-geometric==2.6.1
# uv pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu128.html

uv pip install "numpy<=1.26.0" scipy scikit-learn pandas ase==3.25.0 plotly imageio seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat nglview py3Dmol==2.5.0 hydra-submitit-launcher hydra-core==1.* wandb==0.21.0 pyyaml dxtb[libcint] torchmetrics joblib==1.5.1 submitit rmsd pytorch_warmup e3nn==0.5.1 "huggingface_hub>=0.27.1" "kagglehub>=0.3.12" networkx==3.4.2 pydantic==2.11.4 opt-einsum-fx==0.1.4 lmdb==1.5.1 "h5py>=3.10.0" progressbar==2.5 ruff triton==2.2.0 lightning==2.5.1.post0
# fairchem-core==1.10.0

uv pip install -e .

uv pip install -U "jax[cuda12]"==0.6.2
uv pip install -e sella
uv pip install git+https://github.com/virtualzx-nad/geodesic-interpolate.git

# uv pip install autograd==1.5 dask==2023.5.1 distributed==2023.5.1 fabric==3.1.0 jinja2==3.1.5 natsort==8.3.1 rmsd==1.5.1

uv pip install pyscf
uv pip install gpu4pyscf-cuda12x cutensor-cu12

cd ../ReactBench
cd dependencies 
git clone git@github.com:BurgerAndreas/pysisyphus.git 
cd pysisyphus 
uv pip install -e .
cd ..

git clone git@github.com:BurgerAndreas/pyGSM.git 
cd pyGSM
uv pip install -e .
cd ../..

cd ReactBench/MLIP/leftnet/ # install leftnet env
uv pip install -e .
cd ../../..

cd ReactBench/MLIP/mace/ # install mace env
uv pip install -e .
cd ../../..

# git clone git@github.com:BurgerAndreas/gad-ff.git
cd ../gad-ff
uv pip install -e .
cd ../ReactBench

cd ../gad-ff
```



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
elements = torch.tensor([1, 6, 7, 8]) # H, C, N, O
data = TGData(
    pos=torch.randn(n_atoms, 3),
    batch=torch.zeros(n_atoms),
    z=elements[torch.randint(0, 4, (n_atoms,))],
    natoms=n_atoms,
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