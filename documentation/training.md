# Training HORM EquiformerV2 on Eigen dataset

Extra prediction heads for the Hessian eigenvalues and eigenvectors.

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