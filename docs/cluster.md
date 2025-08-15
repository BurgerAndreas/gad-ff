# Installation

### First time connecting to a cluster

```bash
ssh -Y aburger@killarney.alliancecan.ca
```

Add ssh key to github
```bash
ssh-keygen -t ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

```bash
git config --global user.name "Max Mustermann"
git config --global user.email "max.mustermann@mail.utoronto.ca"
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

# install the rest
pip3 uninstall torch-cluster pyg-lib torch-scatter torch-sparse torch-geometric -y
pip3 install --no-cache-dir --no-index torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu128.html
pip3 install --no-cache-dir torch-geometric==2.6.1

pip3 install --no-cache-dir numpy<=1.26.0 scipy scikit-learn pandas ase==3.25.0 plotly imageio seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat nglview py3Dmol==2.5.0 hydra-submitit-launcher hydra-core==1.* wandb==0.19.11w pyyaml dxtb[libcint] torchmetrics joblib submitit rmsd pytorch_warmup e3nn==0.5.1 huggingface_hub>=0.27.1 kagglehub>=0.3.12 networkx==3.4.2 pydantic==2.11.4 opt-einsum-fx==0.1.4 lmdb==1.5.1 h5py>=3.10.0 progressbar==2.5
# fairchem-core==1.10.0
pip3 install --no-cache-dir triton==2.2.0 lightning==2.5.1.post0
# pip3 install les@git+https://github.com/ChengUCB/les

# compute canada
# pip3 install --no-cache-dir rdkit==2024.9.6
module load rdkit/2023.09.5 openmm/8.2.0 openbabel/3.1.1 mctc-lib/0.3.1

pip install -e .

pip install -U "jax[cuda12]"==0.6.2
pip install -e sella
pip install quacc
pip install git+https://github.com/virtualzx-nad/geodesic-interpolate.git

pip install autograd==1.5 dask==2023.5.1 distributed==2023.5.1 fabric==3.1.0 jinja2==3.1.5 natsort==8.3.1 rmsd==1.5.1


pip3 install pyscf==2.2.1 
pip3 install gpu4pyscf-cuda12x cutensor-cu12
pip3 install cupy-cuda12x==13.4.1

mamba install xtb-python -c conda-forge

```

```bash
cd ..
git clone https://gitlab.com/matschreiner/Transition1x
cd Transition1x
pip install .
python download_t1x.py
mv data/transition1x.h5 ../gad-ff/data/transition1x.h5
cd ../gad-ff
```

I had problems with the compute canada version of wandb, so I installed it manually
```bash
pip uninstall wandb -y

wget https://files.pythonhosted.org/packages/88/c9/41b8bdb493e5eda32b502bc1cc49d539335a92cacaf0ef304d7dae0240aa/wandb-0.20.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -O wandb-0.20.1-py3-none-any.whl

PIP_CONFIG_FILE=/dev/null pip3 install wandb-0.20.1-py3-none-any.whl --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-index
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


### Setup

```bash
# download HORM dataset (11GB)
python scripts_horm/download_horm_data_kaggle.py
```

```bash
# Download HORM EquiformerV2 with Energy-Force-Hessian Training
mkdir -p ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2.ckpt -O ckpt/eqv2.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left-df.ckpt -O ckpt/left-df.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left.ckpt -O ckpt/left.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/alpha.ckpt -O ckpt/alpha.ckpt
```


### Start an interactive session

Killarney
```bash
salloc -A aip-aspuru -t 60:00:00 -D /project/aip-aspuru/aburger/gad-ff --gres=gpu:l40s:1 --mem=128GB
```
Balam
```bash
debugjob --clean -g 1
```