

Create a .env file in the root directory and set these variables (adjust as needed):
```
HOMEROOT=${HOME}/gad-ff
# some scratch space where we can write files during training. can be the same as HOMEROOT
PROJECTROOT=${PROJECT}/gad-ff
# the python environment to use (run `which python` to find it)
PYTHONBIN=${HOME}/miniforge3/envs/gad/bin
WANDB_ENTITY=...
MPLCONFIGDIR=${PROJECTROOT}/.matplotlib
```


```bash
module load cuda/12.3
module load gcc/12.3.0
```

```bash
debugjob --clean -g 1
```


```bash
rm -rf ${PROJECT}/.cache
rm -rf ${HOME}/.cache
mkdir -p ${PROJECT}/.cache
# the fake folder we will use
ln -s ${PROJECT}/.cache ${HOME}/.cache
# Check the simlink
ls -la ${PROJECT}/.cache

rm -rf ${PROJECT}/.conda
rm -rf ${HOME}/.conda
mkdir -p ${PROJECT}/.conda
# the fake folder we will use
ln -s ${PROJECT}/.conda ${HOME}/.conda
# Check the simlink
ls -la ${PROJECT}/.conda

rm -rf ${PROJECT}/.mamba
rm -rf ${HOME}/.mamba
mkdir -p ${PROJECT}/.mamba
# the fake folder we will use
ln -s ${PROJECT}/.mamba ${HOME}/.mamba
# Check the simlink
ls -la ${PROJECT}/.mamba
```