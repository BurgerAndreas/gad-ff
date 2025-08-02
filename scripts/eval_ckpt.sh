#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/aburger/gad-ff
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:l40s:1 
#SBATCH --mem=128GB
#SBATCH --job-name=eval 
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/project/aip-aspuru/aburger/gad-ff/outslurm/slurm-%j.txt 
#SBATCH --error=/project/aip-aspuru/aburger/gad-ff/outslurm/slurm-%j.txt

REACT_DIR=/project/aip-aspuru/aburger/ReactBench
GAD_DIR=/project/aip-aspuru/aburger/gad-ff

gad
mkdir -p outslurm

source gadenv/bin/activate
module load cuda/12.6
module load gcc/12.3
module load rdkit/2023.09.5 openmm/8.2.0 openbabel/3.1.1 mctc-lib/0.3.1

CKPT=$(python ${GAD_DIR}/scripts/get_ckpt_name.py $@ | tail -n 1)

echo "Using checkpoint: ${CKPT}"

mkdir -p ${GAD_DIR}/ckpteval
cp ${CKPT} ${GAD_DIR}/ckpteval/

sbatch ${GAD_DIR}/scripts/killarney.sh ${GAD_DIR}/scripts/eval_horm.py --checkpoint=${CKPT} --dataset=ts1x-val.lmdb

react
sbatch ${REACT_DIR}/killarney.sh ${REACT_DIR}/ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=${CKPT}
