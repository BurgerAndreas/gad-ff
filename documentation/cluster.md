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
pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# check this works
gadenv/bin/python
import torch
exit()

# install the rest
pip3 install --no-cache-dir --no-index pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip3 install --no-cache-dir torch-geometric

pip3 install --no-cache-dir numpy>=1.26.0 scipy scikit-learn pandas ase==3.25.0 plotly imageio seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat nglview py3Dmol==2.5.0 hydra-submitit-launcher hydra-core==1.* wandb==0.19.11 pyyaml dxtb[libcint] torchmetrics joblib submitit rmsd pytorch_warmup e3nn==0.5.1 fairchem-core==1.10.0 huggingface_hub>=0.27.1 networkx==3.4.2 pydantic==2.11.4 opt-einsum-fx==0.1.4 lmdb==1.5.1 h5py>=3.10.0 progressbar==2.5
pip3 install --no-cache-dir triton==2.2.0 pytorch-lightning==2.5.1.post0
pip3 install --no-cache-dir rdkit==2024.9.6
pip3 install les@git+https://github.com/ChengUCB/les

# if you are using mace
# pip3 install --no-cache-dir cuequivariance-ops-torch-cu12==0.5.0 cuequivariance-torch==0.5.0 

# mamba install openmm==8.2.0 openbabel=3.1.1
```

rdkit installation failed on compute canada, so we force it
```bash
# download directly from 
# https://pypi.org/project/rdkit/2024.9.6/#rdkit-2024.9.6-cp311-cp311-manylinux_2_28_x86_64.whl
wget https://files.pythonhosted.org/packages/cc/3f/472c33312ca8a55242fc2cf6179809f4a967185e9dc6e76ea28ddd37a097/rdkit-2024.9.6-cp311-cp311-manylinux_2_28_x86_64.whl
# rename to bypass 'is not a supported wheel on this platform'
cp rdkit-2024.9.6-cp311-cp311-manylinux_2_28_x86_64.whl rdkit-2024.9.6-py3-none-any.whl

# overwrite any kinds of compute canada restrictions
PIP_CONFIG_FILE=/dev/null pip3 install rdkit-2024.9.6-py3-none-any.whl --force-reinstall --no-deps --no-build-isolation --no-cache-dir --find-links "" --constraint /dev/null --no-index --no-warn-script-location --disable-pip-version-check
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

### Setup

```
# download HORM dataset (11GB)
python scripts_horm/download_horm_data_kaggle.py
```
