# Training HORM EquiformerV2 on Eigen dataset

Extra prediction heads for the Hessian eigenvalues and eigenvectors.

## Start job

Killarney
```bash
salloc -A aip-aspuru -t 60:00:00 -D /project/aip-aspuru/aburger/gad-ff --gres=gpu:l40s:1 --mem=128GB
source gadenv/bin/activate
source .env
```

## Training

Verify the training script and environment are working
```bash
python scripts/train_eigen.py experiment=debug
sbatch scripts/killarney.sh scripts/train_eigen.py experiment=debug
```

Verify that we can overfit to the tiny dataset (using one L40s)
```bash
sbatch scripts/killarney.sh scripts/train_eigen.py experiment=overfit100
```

Fit on the smaller training dataset (RGD1), test on TS1x
```bash
sbatch scripts/killarney_2xl40s.sh scripts/train_eigen.py experiment=rgd1 gpu=two
sbatch scripts/killarney_2xl40s.sh scripts/train_eigen.py experiment=rgd1 gpu=two training.lr_schedule_type=null

# smaller batch size instead
sbatch scripts/killarney.sh scripts/train_eigen.py experiment=rgd1 training.bz=100
sbatch scripts/killarney.sh scripts/train_eigen.py experiment=rgd1 training.bz=100 training.lr_schedule_type=null

# H100 instead
sbatch scripts/killarney_h100.sh scripts/train_eigen.py experiment=rgd1 
```

Fit on the larger training dataset (TS1x), test on TS1x
```bash
sbatch scripts/killarney.sh scripts/train_eigen.py experiment=ts1x
```

Fit on both datasets (RGD1 and TS1x), test on TS1x
```bash
sbatch scripts/killarney_h100.sh scripts/train_eigen.py experiment=alldata
```

## Background

We have two training datasets and one validation dataset:
```bash
ls ~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5

total 23G
1.1G RGD1.lmdb (60000 samples)
21G ts1x_hess_train_big.lmdb (1725362 samples)
620M ts1x-val.lmdb (50844 samples)
```
Plus a tiny subset for debugging
```bash
ls data

sample_100.lmdb
```

There are two kinds of nodes on our Killarney cluster:

Performance Tier | Nodes | Model | CPU | Cores | System Memory | GPUs per node | Total GPUs \
Standard Compute | 168 | Dell 750xa | 2 x Intel Xeon Gold 6338 | 64 | 512 GB | 4 x NVIDIA L40S 48GB | 672 \
Performance Compute | 10 | Dell XE9680 | 2 x Intel Xeon Gold 6442Y | 48 | 2048 GB | 8 x NVIDIA H100 SXM 80GB | 80


