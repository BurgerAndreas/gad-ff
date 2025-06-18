from ase import build
import torch
import numpy as np

from typing import Optional, Sequence
import torch.utils.data

# MACE-OFF23	10	SPICE v1	DFT (wB97M+D3)	Organic Chemistry
# https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model
from mace.calculators import mace_off
from mace.data import AtomicData
from mace.data.utils import config_from_atoms, KeySpecification
from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

# Load MACE-OFF model
model = mace_off(
    model="medium", 
    # enable_cueq=True, 
    default_dtype="float64", device='cuda',
    return_raw_model=True
)

print(f"Loaded MACE model: {type(model)}")

def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

