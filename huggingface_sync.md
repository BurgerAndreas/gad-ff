
sync to huggingface
```bash
# find the checkpoint you want to sync
python gadff/find_checkpoint.py wandb_run_id gadff

# get checkpoint
cp checkpoints/mycheckpoint ckpt/

cd ../heigen 

# rsync -av --exclude='__pycache__' ../gad-ff/ocpmodels ./
# rsync -av --exclude='__pycache__' ../gad-ff/nets ./


cp ../gad-ff/ckpt/hesspred_v1.ckpt ./ckpt/ # hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746
cp ../gad-ff/ckpt/hesspred_v2.ckpt ./ckpt/ # 
cp ../gad-ff/ckpt/hesspred_v3.ckpt ./ckpt/

# cp ../gad-ff/scripts/download_horm_data_kaggle.py ./

mkdir data
cp ../gad-ff/data/sample_100.lmdb ./data/

# upload to huggingface
huggingface-cli upload andreasburger/heigen .

cd ../gad-ff
```