# pip install datasets (from HuggingFace)
from datasets import load_from_disk, DatasetDict

# https://figshare.com/articles/dataset/_b_Hessian_QM9_Dataset_b_/26363959?file=49271011
dataset = load_from_disk("/ssd/Code/Datastore/qm9hessian/hessian_qm9_DatasetDict")
print("Original dataset:")
print(dataset)

# Split each dataset into train and test
split_dataset = DatasetDict()
for split_name, ds in dataset.items():
    split_ds = ds.train_test_split(test_size=0.1, seed=42)
    split_dataset[f"{split_name}_train"] = split_ds["train"]
    split_dataset[f"{split_name}_test"] = split_ds["test"]
    print(f"\n{split_name}:")
    print(f"  Train: {len(split_ds['train'])} samples")
    print(f"  Test: {len(split_ds['test'])} samples")

print("\nSplit dataset structure:")
print(split_dataset)
# DatasetDict({
#     vacuum: Dataset({
#         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
#         num_rows: 41645
#     })
#     thf: Dataset({
#         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
#         num_rows: 41645
#     })
#     toluene: Dataset({
#         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
#         num_rows: 41645
#     })
#     water: Dataset({
#         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
#         num_rows: 41645
#     })
# })

# thf  toluene  vacuum  water

import jax.numpy as jnp

# https://github.com/google-research/e3x
params = jnp.load(
    "/ssd/Code/Datastore/qm9hessian/model_params/params_train_f128_i5_b16.npz",
    allow_pickle=True,
)["params"].item()
