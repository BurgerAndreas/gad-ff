from typing import List, Optional, Tuple

import os
import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn
import torch.utils.data

from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props

from e3nn import o3
from ocpmodels.hessian_graph_transform import HessianGraphTransform

skip_keys = [
    "pos",
    "rxn",
    "charges",
    "forces",
    "energy",
    "one_hot",
    "cell_offsets",
    "cell_offset_distances",
    "neighbors",
    "ae",
    "hessian",
]


def get_shape(a):
    if isinstance(a, torch.Tensor):
        return list(a.shape)
    elif isinstance(a, np.ndarray):
        return list(a.shape)
    elif isinstance(a, list):
        return [get_shape(x) for x in a]
    else:
        return None


def myprint(batch):
    for k, v in batch.items():
        if k in skip_keys:
            continue
        print(f"{k}   : {get_shape(v)}")
        if v.numel() > 1:
            if v[0].numel() > 5:
                print(f"{k} 0 : {v[0][:5]}")
                print(f"{k} -1: {v[-1][:5]}")
            else:
                print(f"{k} 0 : {v[0]}")
                print(f"{k} -1: {v[-1]}")
        else:
            print(f"{k} 0 : {v.item()}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))

    config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    model_config["do_hessian"] = True
    model_config["otf_graph"] = False
    model = EquiformerV2_OC20(**model_config)

    checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
    state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
    state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to("cuda")

    from ocpmodels.ff_lmdb import LmdbDataset

    dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset = LmdbDataset(dataset_path, transform=transform)

    # print shape and first element of each attribute
    print("\nDataset sample 1")
    sample = dataset[1]
    myprint(sample)

    # will add <>_ptr and <>_batch to the data object
    follow_batch = ["diag_ij", "edge_index", "message_idx_ij"]

    # print("\nSingle sample")
    # dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False, follow_batch=follow_batch)
    # for batch_base in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
    #     N = batch_base.natoms.sum().item()
    #     B = batch_base.batch.max().item() + 1

    #     # print shape and first element of each attribute
    #     print(f"\nDataloader B={B}")
    #     myprint(batch_base)
    #     break

    print("\nBatching")
    _batch_saved = None
    dataloader = TGDataLoader(
        dataset, batch_size=2, shuffle=False, follow_batch=follow_batch
    )
    for batch_base in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
        N = batch_base.natoms.sum().item()
        B = batch_base.batch.max().item() + 1

        # print shape and first element of each attribute
        print(f"\nDataloader B={B}")
        myprint(batch_base)
        _batch_saved = batch_base
        break

    # compare sample and _batch_saved
    print("\nComparing sample and _batch_saved")
    for k, v in sample.items():
        if k in skip_keys:
            continue
        vb = _batch_saved[k]
        print(f"{k}    : {get_shape(v)} | {get_shape(vb)} (B)")
        if vb.numel() > 1:
            if vb[0].numel() > 5:
                print(f"{k} S 0 : {v[0][:5]}")
                print(f"{k} B 0 : {vb[0][:5]}")
                print(f"{k} S -1: {v[-1][:5]}")
                print(f"{k} B -1: {vb[-1][:5]}")
            else:
                v0 = v[0] if v.numel() > 1 else v.item()
                print(f"{k} S 0 : {v0}")
                print(f"{k} B 0 : {vb[0]}")
                if v.numel() > 1:
                    print(f"{k} S -1: {v[-1]}")
                print(f"{k} B -1: {vb[-1]}")
        else:
            print(f"{k} S 0 : {v.item()} | {vb.item()} (B)")
            print(f"{k} S -1: {v.item()} | {vb.item()} (B)")
