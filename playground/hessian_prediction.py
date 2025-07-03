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

from nets.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from nets.equiformer_v2.module_list import ModuleListInfo
from nets.equiformer_v2.so2_ops import SO2_Convolution
from nets.equiformer_v2.radial_function import RadialFunction
from nets.equiformer_v2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from nets.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)



if __name__ == "__main__":

    project_root = os.path.dirname(os.path.dirname(__file__))

    config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    model = EquiformerV2_OC20(**model_config)

    checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
    state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
    state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model.train()
    model.to("cuda")

    # Example 1: load a dataset file and predict the first batch
    from ocpmodels.ff_lmdb import LmdbDataset

    dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    dataset = LmdbDataset(dataset_path)

    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)


    print("\n")
    for batch in tqdm(dataloader, desc='Evaluating', total=len(dataloader)):
        
        batch = batch.to(model.device)
        batch = compute_extra_props(batch, pos_require_grad=True)
        energy, forces, out = model.forward(batch, eigen=True, hessian=True)
        
        pred_hessian = out["hessian"]
        
        print(batch.keys())
        true_hessian = batch.hessian
        
        # compute loss
        loss = torch.nn.functional.mse_loss(pred_hessian, true_hessian)
        
        # backprop
        loss.backward()
        
        break