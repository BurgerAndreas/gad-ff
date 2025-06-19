# HORM：A Molecular Hessian Database for Optimizing Reactive Machine Learning Interatomic Potentials

This is the official implementation for the paper: "A Molecular Hessian Database for Optimizing Reactive Machine Learning Interatomic Potentials". 

Paper: https://arxiv.org/abs/2505.12447
Code: https://github.com/deepprinciple/HORM


- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [How to run this code](#how-to-run-this-code)

# Overview
Transition state (TS) characterization is central to computational reaction modeling, yet conventional approaches depend on expensive density functional theory (DFT) calculations, limiting their scalability. Machine learning interatomic potentials (MLIPs) have emerged as a promising approach to accelerate TS searches by approximating quantum-level accuracy at a fraction of the cost. However, most MLIPs are primarily designed for energy and force prediction, thus their capacity to accurately estimate Hessians, which are crucial for TS optimization, remains constrained by limited training data and inadequate learning strategies. This work introduces the Hessian dataset for Optimizing Reactive MLIP (HORM), the largest quantum chemistry Hessian database dedicated to reactive systems, comprising 1.84 million Hessian matrices computed at the $\omega$B97X/6-31G(d) level of theory. To effectively leverage this dataset, we adopt a Hessian-informed training strategy that incorporates stochastic row sampling, which addresses the dramatically increased cost and complexity of incorporating second-order information into MLIPs. Various MLIP architectures and force prediction schemes trained on HORM demonstrate up to a 63\% reduction in Hessian mean absolute error and up to a 200× increase in TS search success rates compared to models trained without Hessian information. These results highlight how HORM addresses critical data and methodological gaps, enabling the development of more accurate and robust reactive MLIPs for large-scale reaction network exploration.


# Installation Guide:


```shell
pip install torch==2.2.1
pip install . # building wheel might take a while
```

For torch-cluster installation, you need to install the version that matches your CUDA version. 
For example, if you encounter CUDA-related errors, you can uninstall torch-cluster and install the version matching your CUDA version. For CUDA 12.1:

```shell
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```



# How to run this code:

### Train models

To train a model, please select the desired architecture from the available options: 'LeftNet', 'EquiformerV2', and 'AlphaNet'.  
Specify your choice in the `model_type` field within the `train.py` file.
```shell
python train.py
```

### Evaluate models

To evaluate a model, please specify the lmdb dataset and checkpoint, and run the following command:
```shell
python eval.py
```

EquiformerV2 MAE should be:
Transition1x:
Energy: 0.02 eV
Force: 0.02 eV/A
Hessian: 0.08 eV/A^2
Hessian eigenvalues: 0.003 eV/A^2
RGD1:
Energy: 0.13 eV
Force: 0.05 eV/A
Hessian: 0.09 eV/A^2
Hessian eigenvalues: 0.003 eV/A^2

## Dataset

The HORM dataset is available at: https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/data

Run the following command to download the dataset:
```bash
# default is ~/.cache
# export KAGGLEHUB_CACHE=/path/to/your/preferred/directory
python download_horm_data_kaggle.py
```

## Model checkpoints
Pre-trained model checkpoints can be downloaded from: https://huggingface.co/yhong55/HORM

Checkpoints available:

Filename | Model | Training Method \
alpha_orig.ckpt | AlphaNet | Energy-Force Training \
alpha.ckpt | AlphaNet | Energy-Force-Hessian Training \
left_orig.ckpt | LEFTNet | Energy-Force Training \
left.ckpt | LEFTNet | Energy-Force-Hessian Training \
left-df_orig.ckpt | LEFTNet-df | Energy-Force Training \
left-df.ckpt | LEFTNet-df | Energy-Force-Hessian Training \
eqv2_orig.ckpt | EquiformerV2 | Energy-Force Training \
eqv2.ckpt | EquiformerV2 | Energy-Force-Hessian Training \


To download specific model checkpoints, use the following command:
```shell
# Download EquiformerV2 with Energy-Force-Hessian Training
mkdir -p ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2.ckpt -O ckpt/eqv2.ckpt
```


# Notes

### Dataset
- The models were trained on the HORM dataset.
- The geometries in the HORM dataset are sampled from Transition1x and RGD1. 
- All sampled geometries were recomputed by GPU4PYSCF at the ω B97X/6-31G* level of theory to obtain energies, forces, and Hessian
- T1x: 9,000 reactions to the training set and the remaining 1,073 reactions to the validation set. From these, 1,725,362 geometries corresponding to the training reactions and 50,844 geometries from the validation reactions were used to train the models.
- RGD1: From approximately 950,000 available reactions, we randomly selected 80,000 and sampled up to 15 geometries per reaction along their IRC trajectories. From this pool, 60,000 geometries were randomly chosen.

### Training
- The Hessian was computed from the forces via autograd.
- Randomly sample a subset of two columns from each Hessian matrix during training.
- Training was done on A30 (24 GB) and H20 (96 GB) GPUs

For EquiformerV2:

Layers HiddenDim Heads NHR LearningRate BatchSize \
4 128 4 2 3e-4 128








# License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). For more details, please refer to the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
