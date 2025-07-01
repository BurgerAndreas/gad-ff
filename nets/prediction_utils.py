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
    if hasattr(batch, "one_hot"):
        # this is only for the HORM dataset
        # it uses a weird convention
        # atom types are encoded as one-hot vectors of shape (N, 5)
        # where the fifth is unused, likely a padding or None class
        # corresponds to H, C, N, O, None
        indices = batch.one_hot.long().argmax(dim=1)
        batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
    elif hasattr(batch, "z"):
        batch.z = batch.z.to(device)
    else:
        raise ValueError("batch has no one_hot or z attribute")
    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    # atomization energy. shape used by equiformerv2
    if not hasattr(batch, "ae"):
        batch.ae = torch.tensor(0.0, device=device, dtype=torch.float64)
    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch
