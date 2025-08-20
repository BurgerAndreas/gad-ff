# GAD-FF

Directly predict the lowest Hessian eigenvalues and their eigenvectors for fast transition state search.
Based on the HORM EquiformerV2 model, finetuned with extra prediction heads on the HORM DFT Hessians.

## Installation

### Setting up the environment
Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
git clone git@github.com:BurgerAndreas/gad-ff.git
cd gad-ff

uv venv .hessenv --python 3.11
source .hessenv/bin/activate
uv pip install --upgrade pip

uv pip install torch==2.7.0  --index-url https://download.pytorch.org/whl/cu126
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-geometric

uv pip install -r requirements.txt
# fairchem-core==1.10.0

uv pip install -e .

uv pip install -U "jax[cuda12]"==0.6.2
uv pip install -e sella
uv pip install git+https://github.com/virtualzx-nad/geodesic-interpolate.git

uv pip install gpu4pyscf-cuda12x cutensor-cu12
```

To run the evals you need the sister repository as well:
```bash
cd ..
git clone git@github.com:BurgerAndreas/ReactBench.git
cd ReactBench

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

uv pip install -e . # install ReactBench

# Get the recomputed Transition1x subset for validation, 960 datapoints
mkdir -p data 
tar -xzf ts1x.tar.gz -C data
find data/ts1x -type f | wc -l # 960

cd ../gad-ff
```

### Setting up the dataset
Kaggle automatically downloads to the `~/.cache` folder. I highly recommend to set up a symbolic link to a local folder to avoid running out of space:
```bash
PROJECT = <folder where you want to store the dataset>
mkdir -p ${PROJECT}/.cache
ln -s ${PROJECT}/.cache ${HOME}/.cache
```

Get the HORM dataset: # TODO: upload preprocessed data
```bash
python scripts/download_horm_data_kaggle.py
```

Preprocess the Hessian dataset (takes ~48 hours) 
```bash
python scripts/preprocess_hessian_dataset.py --dataset-file data/sample_100.lmdb

python scripts/preprocess_hessian_dataset.py --dataset-file ts1x-val.lmdb
python scripts/preprocess_hessian_dataset.py --dataset-file RGD1.lmdb
python scripts/preprocess_hessian_dataset.py --dataset-file ts1x_hess_train_big.lmdb
```

### Get model checkpoints

Get the baseline model:
- `ckpt/eqv2.ckpt`: HORM EquiformerV2 finetuned on the HORM Hessian dataset. Can be used to get the Hessian via autograd. Used as starting point for training our HessianLearning model as well as baseline for evaluation.

```bash
# Download HORM EquiformerV2 with Energy-Force-Hessian Training
mkdir -p ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2.ckpt -O ckpt/eqv2.ckpt
# Other models from the HORM paper
wget https://huggingface.co/yhong55/HORM/resolve/main/left-df.ckpt -O ckpt/left-df.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left.ckpt -O ckpt/left.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/alpha.ckpt -O ckpt/alpha.ckpt
```

Get our HessianLearning model: # TODO
```bash
# download from HuggingFace
wget https://huggingface.co/andreasburger/heigen -O ckpt/heigen.ckpt
```


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


## Reproduce results from our paper

Training run we used: # TODO
```bash

```

Evaluation: # TODO



## Citation

If you found this code useful, please consider citing:
```bibtex
TODO
```

The training code and the dataset are based on the HORM [paper](https://arxiv.org/abs/2505.12447), [dataset](https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/data), and [code](https://github.com/deepprinciple/HORM)
```bibtex
HORM
```

The evaluation is based on the ReactBench [paper](https://chemrxiv.org/engage/chemrxiv/article-details/68270569927d1c2e66165ad8), and [code](https://github.com/deepprinciple/ReactBench)
```bibtex
@article{https://doi.org/10.1002/advs.202506240,
    author = {Zhao, Qiyuan and Han, Yunhong and Zhang, Duo and Wang, Jiaxu and Zhong, Peichen and Cui, Taoyong and Yin, Bangchen and Cao, Yirui and Jia, Haojun and Duan, Chenru},
    title = {Harnessing Machine Learning to Enhance Transition State Search with Interatomic Potentials and Generative Models},
    journal = {Advanced Science},
    pages = {e06240},
    doi = {https://doi.org/10.1002/advs.202506240}
}
```

We thank the authors of HORM and ReactBench from DeepPrinciple for making their code and data openly available. Please consider citing their work if you use this code or data.