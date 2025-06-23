# Installation

### First time connecting to a cluster

Add ssh key to github
```bash
ssh-keygen -t ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

### Python env

```bash
python3.11 -m venv gadenv
source gadenv/bin/activate
```

```bash
# module spider cuda
module load cuda/12.6
module load gcc/12.3
```

```bash
pip3 uninstall torch torchvision torchaudio torch-scatter torch-sparse torch-cluster torch-spline-conv -y
pip3 install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu126

# check this works
gadenv/bin/python
import torch
exit()

# install the rest
pip3 uninstall torch-cluster pyg-lib torch-scatter torch-sparse torch-geometric -y
pip3 install --no-cache-dir --no-index torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu128.html
pip3 install --no-cache-dir torch-geometric==2.6.1

pip3 install --no-cache-dir numpy>=1.26.0 scipy scikit-learn pandas ase==3.25.0 plotly imageio seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat nglview py3Dmol==2.5.0 hydra-submitit-launcher hydra-core==1.* wandb==0.19.11 pyyaml dxtb[libcint] torchmetrics joblib submitit rmsd pytorch_warmup e3nn==0.5.1 huggingface_hub>=0.27.1 kagglehub>=0.3.12 networkx==3.4.2 pydantic==2.11.4 opt-einsum-fx==0.1.4 lmdb==1.5.1 h5py>=3.10.0 progressbar==2.5
#  fairchem-core==1.10.0
pip3 install --no-cache-dir triton==2.2.0 pytorch-lightning==2.5.1.post0
pip3 install les@git+https://github.com/ChengUCB/les

# compute canada
# pip3 install --no-cache-dir rdkit==2024.9.6
module load rdkit/2023.09.5 openmm/8.2.0 openbabel/3.1.1 mctc-lib/0.3.1

pip install -e .
```

If you want to use MACE
```bash
pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
git clone https://github.com/ACEsuit/mace.git
pip install -e ./mace
```


### Setup
Create a .env file in the root directory and set these variables (adjust as needed):
```bash
touch .env
nano .env
```
```bash
# .env
HOMEROOT=${HOME}/gad-ff
# some scratch space where we can write files during training. can be the same as HOMEROOT
PROJECTROOT=${PROJECT}/gad-ff
# the python environment to use (run `which python` to find it)
PYTHONBIN=${HOME}/gad-ff/gadenv/bin/python
WANDB_ENTITY=...
MPLCONFIGDIR=${PROJECTROOT}/.matplotlib
```


Simlink
```bash
rm -rf ${PROJECT}/.cache
rm -rf ${HOME}/.cache
mkdir -p ${PROJECT}/.cache
# the fake folder we will use
ln -s ${PROJECT}/.cache ${HOME}/.cache
# Check the simlink
ls -la ${PROJECT}/.cache

rm -rf ${PROJECT}/.conda
rm -rf ${HOME}/.conda
mkdir -p ${PROJECT}/.conda
# the fake folder we will use
ln -s ${PROJECT}/.conda ${HOME}/.conda
# Check the simlink
ls -la ${PROJECT}/.conda

rm -rf ${PROJECT}/.mamba
rm -rf ${HOME}/.mamba
mkdir -p ${PROJECT}/.mamba
# the fake folder we will use
ln -s ${PROJECT}/.mamba ${HOME}/.mamba
# Check the simlink
ls -la ${PROJECT}/.mamba
```

### Start job

Killarney
```bash
salloc -A aip-aspuru -t 60:00:00 -D /project/aip-aspuru/aburger/gad-ff --gres=gpu:l40s:1 --mem=128GB
```
Balam
```bash
debugjob --clean -g 1
```

Try training
```bash
python scripts/train_eigen.py +extra=debug
```

### Setup

```shell
# download HORM dataset (11GB)
python scripts_horm/download_horm_data_kaggle.py
```

```shell
# Download EquiformerV2 with Energy-Force-Hessian Training
mkdir -p ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2.ckpt -O ckpt/eqv2.ckpt
```