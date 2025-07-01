
sync to huggingface
```bash
# find the checkpoint you want to sync
python gadff/find_checkpoint.py wandb_run_id gadff

# get checkpoint
cp checkpoints/mycheckpoint ckpt/


cd ../heigen 


# remove all files except ckpt
find . -mindepth 1 ! -name 'ckpt' ! -path './ckpt/*' -exec rm -rf {} +

# copy over files
cp ../gad-ff/gadff/equiformer_ase_calculator.py ./
cp ../gad-ff/playground/example_inference.py ./

rsync -av --exclude='__pycache__' ../gad-ff/ocpmodels ./
rsync -av --exclude='__pycache__' ../gad-ff/nets ./

cp -r ../gad-ff/ckpt ./

# cp ../gad-ff/scripts/download_horm_data_kaggle.py ./

cp -r ../gad-ff/docs ./
cp ../gad-ff/readme.md ./

mkdir data
cp ../gad-ff/data/ff_lmdb.py ./data/
cp ../gad-ff/data/sample_100-dft-hess-eigen.lmdb ./data/

mkdir configs
cp ../gad-ff/configs/equiformer_v2.yaml ./configs/

# upload to huggingface
huggingface-cli upload andreasburger/heigen .

cd ../gad-ff
```