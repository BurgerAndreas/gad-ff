#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sampler 
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/project/a/aspuru/aburger/gad-ff/outslurm/slurm-%j.txt 
#SBATCH --error=/project/a/aspuru/aburger/gad-ff/outslurm/slurm-%j.txt

# get environment variables
source .env

module load cuda/12.3
module load gcc/12.3.0

# activate environment
# source ${PYTHONBIN}/activate

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "Inside slurm_launcher.slrm ($0). received arguments: $@"

# hand over all arguments to the script
echo "Submitting ${HOMEROOT}/$@"

${PYTHONBIN}/python ${HOMEROOT}/"$@"