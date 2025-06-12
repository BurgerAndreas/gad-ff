

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

```bash
git clone git@github.com:BurgerAndreas/gad-ff.git
cd gad-ff
```


Create a new conda environment and activate it
```bash
mamba create --name gad python=3.12 -y
mamba activate gad
```

Install packages
```bash
# setuptools==59.2.0
pip install numpy scikit-learn pandas ase plotly kaleido imageio scipy matplotlib seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat nglview py3Dmol hydra-core==1.* wandb
pip install les@git+https://github.com/ChengUCB/les
mamba install openmm==8.2.0 openbabel=3.1.1 conda-forge::rdkit=2024 -y

# pip install torch
# Install PyTorch for CUDA 12.1
pip uninstall torch torchvision torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-index pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric

pip install -e .
```

Install NewtonNet
```bash
# git clone https://github.com/THGLab/NewtonNet/tree/v2.0.1

cd NewtonNet

pip install -e .
cd ..
```

Download ANI-1x dataset
[paper](https://pubs.aip.org/aip/jcp/article/148/24/241733/963478/Less-is-more-Sampling-chemical-space-with-active)
[paper](https://www.nature.com/articles/s41597-020-0473-z)
[example scripts](https://github.com/aiqm/ANI1x_datasets)
[dataset](https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041?backTo=%2Fcollections%2FThe_ANI-1ccx_and_ANI-1x_data_sets_coupled-cluster_and_density_functional_theory_properties_for_molecules%2F4712477&file=18112775)
```bash
mkdir -p Data/ANI1x/data
cd Data/ANI1x
pip install -e .
cd data
# 5 GB
wget https://springernature.figshare.com/ndownloader/files/18112775 -O ani1x-release.h5
cd ../../..
```

Download Transition-1x dataset 
[paper](https://www.nature.com/articles/s41597-022-01870-w)
[example scripts](https://gitlab.com/matschreiner/Transition1x)
```bash
# wget https://figshare.com/ndownloader/files/36035789 -O Transition1x.h5

cd Data
git clone https://gitlab.com/matschreiner/Transition1x
cd Transition1x
pip install -e .
# pip install '.[example]'
# will download to Transition1x/data/Transition1x.h5
python download_t1x.py ./data
cd ../..
```

Process data by running
- `MLHessian-TSopt/Scripts/split.ipynb`

## TODO

Check which model / checkpoint has the nicest Hessian:
- `MLHessian-TSopt/Analysis/Figure2.ipynb` and `MLHessian-TSopt/Analysis/Figure4bc.ipynb`: Wrapper notebook for model testing regarding Hessian predictions.
- `MLHessian-TSopt/Analysis/Figure4c.ipynb`: Wrapper notebook for optimized transition state comparisons.

Generate GAD dataset using one of the following models:
- `MLHessian-TSopt/Models/PretrainedModels/training_1`: Pre-trained model in NewtonNet paper, trained on ANI dataset.
- `MLHessian-TSopt/Models/PretrainedModels/training_9`: Pre-trained model in NewtonNet paper, trained on ANI-1x dataset.
- `MLHessian-TSopt/Models/FinetunedModels/training_44`: Fine-tuned model from `training_1` above, trained on Transition-1x dataset composition split 5.

Finetune NewtonNet using their training script:
- `NewtonNet/scripts/newtonnet_train.py`

Test the model:
- `Scripts/test.ipynb`: Wrapper notebook for model testing using the holdout test reactions in Transition-1x dataset.
- `Scripts/noise.ipynb`: Wrapper notebook for initial guess geometry generation and subsequent noising of Sella benchmark reactions.
- `Scripts/opt/nn_sella_quacc.py`: Wrapper script for NewtonNet-based optimizations.