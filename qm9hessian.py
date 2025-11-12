# pip install datasets (from HuggingFace)
from datasets import load_from_disk

# https://figshare.com/articles/dataset/_b_Hessian_QM9_Dataset_b_/26363959?file=49271011
dataset = load_from_disk("/ssd/Code/Datastore/qm9hessian/hessian_qm9_DatasetDict")
print(dataset)

# thf  toluene  vacuum  water

import jax.numpy as jnp

# https://github.com/google-research/e3x
params = jnp.load(
    "/ssd/Code/Datastore/qm9hessian/model_params/params_train_f128_i5_b16.npz",
    allow_pickle=True,
)["params"].item()
