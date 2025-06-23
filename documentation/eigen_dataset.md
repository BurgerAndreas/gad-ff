# Creating the Eigen dataset 
Smallest Hessian eigenvalues and their eigenvectors

For more information about the HORM dataset, see [documentation/horm.md](documentation/horm.md)

### Prerequisites

Start an interactive job
```bash
salloc -A aip-aspuru -t 60:00:00 -D /project/aip-aspuru/aburger/gad-ff --gres=gpu:l40s:1 --mem=128GB
source gadenv/bin/activate
source .env
```

Download the HORM dataset (11GB)
```bash
python scripts_horm/download_horm_data_kaggle.py
```

### Test
Test if the dataset creation is working
```bash
python scripts/create_equ_hess_eigen_dataset.py --dataset-file data/sample_100.lmdb
python scripts/test_eigen_dataset.py --original-dataset data/sample_100.lmdb
```

### Smaller datasets
Create the smaller datasets (~10-20h each)
```bash
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset.py --dataset-file ts1x-val.lmdb
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset.py --dataset-file RGD1.lmdb

# test
python scripts/test_eigen_dataset.py --original-dataset ts1x-val.lmdb
python scripts/test_eigen_dataset.py --original-dataset RGD1.lmdb
```

### Large datasets

We split the datasets into 10 parts that can run in parallel (~3 days each)
```bash
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 0 --end-idx 172536 --job-id 0
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 172536 --end-idx 345072 --job-id 1
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 345072 --end-idx 517608 --job-id 2
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 517608 --end-idx 690144 --job-id 3
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 690144 --end-idx 862680 --job-id 4
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 862680 --end-idx 1035216 --job-id 5
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 1035216 --end-idx 1207752 --job-id 6
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 1207752 --end-idx 1380288 --job-id 7
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 1380288 --end-idx 1552824 --job-id 8
sbatch scripts/killarney.sh scripts/create_equ_hess_eigen_dataset_split.py --process --dataset ts1x_hess_train_big.lmdb --start-idx 1552824 --end-idx 1725362 --job-id 9
```

After the job is done, we can merge the datasets
```bash
python scripts/create_equ_hess_eigen_dataset_split.py --combine --dataset ts1x_hess_train_big.lmdb
```

Check the files (default location)
```bash
ls -lh ~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/
```

### Next steps

Get training
```bash
python scripts/train_eigen.py +experiment=debug
```