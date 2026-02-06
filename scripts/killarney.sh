#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /scratch/aburger/gad-ff
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:l40s:1 
#SBATCH --mem=128GB
#SBATCH --job-name=gad 
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/scratch/aburger/gad-ff/outslurm/slurm-%j.txt 
#SBATCH --error=/scratch/aburger/gad-ff/outslurm/slurm-%j.txt

# activate venv
source .venv/bin/activate

which python

# get environment variables
# source .env

module load cuda/12.6
module load gcc/12.3
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# append command to slurmlog.txt
echo "sbatch scripts/killarney.sh $@ # $SLURM_JOB_ID" >> slurmlog.txt

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "Inside slurm_launcher.slrm ($0). received arguments: $@"

# hand over all arguments to the script
pwd
echo "Submitting $@"

srun uv run "$@"
