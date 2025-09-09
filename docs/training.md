# Training HORM EquiformerV2 on Hessian prediction

Extra prediction head for the Hessian.

## Preprocessing

Killarney
```bash
salloc -A aip-aspuru -t 60:00:00 -D /project/aip-aspuru/aburger/gad-ff --gres=gpu:l40s:1 --mem=128GB
source gadenv/bin/activate
source .env
```

## Training

Verify the training script and environment are working
```bash
python scripts/train.py experiment=debug
sbatch scripts/killarney.sh scripts/train.py experiment=debug
```

Verify that we can overfit to the tiny dataset (using one L40s)
```bash
sbatch scripts/killarney.sh scripts/train.py experiment=overfit100
sbatch scripts/killarney.sh scripts/train.py experiment=overfit100 training.loss_type_vec=cosine
```

Fit on both datasets (RGD1 and TS1x), test on TS1x
```bash
sbatch scripts/killarney_h100.sh scripts/train.py experiment=alldata
```

## Background

The HORM dataset contains two training datasets and one validation dataset:
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
