import torch
from nets.scatter_utils import scatter_mean

GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8])


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


def compute_extra_props(batch, pos_require_grad=True):
    """Adds device, z, and removes mean batch"""
    device = batch.pos.device
    indices = batch.one_hot.long().argmax(dim=1)
    batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    # atomization energy. shape used by equiformerv2
    if not hasattr(batch, "ae"):
        batch.ae = torch.zeros_like(batch.energy)
    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch
