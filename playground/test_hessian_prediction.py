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
from e3nn import o3
from ocpmodels.hessian_graph_transform import HessianGraphTransform
from nets.equiformer_v2.hessian_pred_utils import run_hessian_tests


def equivariance_test(model, batch_base):

    state_before = model.training
    model.eval()

    N = batch_base.natoms.sum().item()

    # regular forward pass
    batch = batch_base.clone()
    batch = batch.to(model.device)
    batch = compute_extra_props(batch, pos_require_grad=True)
    energy, forces, out = model.forward(batch, eigen=True, hessian=True)
    pred_hessian = out["hessian"]
    pred_hessian = pred_hessian.reshape(N * 3, N * 3)  # necessary for 1d hessian

    # R = o3.wigner_D(
    #     1, torch.tensor([0.3]), torch.tensor([1.8]), torch.tensor([-2.8])
    # )[0]
    # R = R.to(model.device)
    R = torch.tensor(o3.rand_matrix()).to(model.device)  # det(R) = +1

    # rotated batch
    batch = batch_base.clone()
    batch = batch.to(model.device)
    batch.pos = batch.pos @ R
    assert torch.allclose(batch.pos @ R, torch.matmul(batch.pos, R))
    batch = compute_extra_props(batch, pos_require_grad=True)
    energy2, forces2, rotated_out = model.forward(batch, eigen=True, hessian=True)
    hessian2 = rotated_out["hessian"]
    hessian2 = hessian2.reshape(N * 3, N * 3)  # necessary for 1d hessian

    print()
    # energy should be invariant
    diffe = energy - energy2
    print(f"Energy abs diff: {diffe.abs().sum().item():.2e}")
    print(f"Energy rel diff: {(diffe.abs() / energy.abs()).mean().item():.2e}")

    # forces should be equivariant
    difff = forces - (forces2 @ R.T)
    print(f"Forces abs diff: {difff.abs().mean().item():.2e}")
    print(f"Forces rel diff: {(difff.abs() / forces.abs()).mean().item():.2e}")

    # hessian should be equivariant
    R_hessian = torch.kron(torch.eye(N, device=model.device, dtype=model.dtype), R)
    diffh = pred_hessian - (R_hessian @ hessian2 @ R_hessian.T)
    print(f"Hessian abs diff: {diffh.abs().mean().item():.2e}")
    _pred_hessian = torch.where(pred_hessian == 0, 1, pred_hessian)
    print(f"Hessian rel diff: {(diffh.abs() / _pred_hessian.abs()).mean().item():.2e}")

    model.train(state_before)


def compute_loss_blockdiagonal_hessian(pred_hessian, true_hessian, loss_fn, data):
    """
    pred_hessian: (B*N, 3, B*N, 3)
    true_hessian: (B*N*3*N*3)
    """
    B = data.batch.max().item() + 1
    N = data.natoms.sum().item()
    pred_hessian = pred_hessian.reshape(N, 3, N, 3)
    loss = 0
    # compare by extracting blocks from full hessian
    atom_offset = 0
    past_entries = 0
    test_mask = torch.zeros_like(pred_hessian)
    for b in range(B):
        n_atoms_batch = data.natoms[b].item()
        assert (
            atom_offset == data.ptr[b].item()
        ), f"Atom offset {atom_offset} does not match batch {b} ptr {data.ptr[b].item()}"
        # Extract the block from the full hessian
        block_start = atom_offset
        block_end = atom_offset + n_atoms_batch
        pred_block = pred_hessian[block_start:block_end, :, block_start:block_end, :]
        test_mask[block_start:block_end, :, block_start:block_end, :] = 1
        # from the flat block
        n_entries_batch = (n_atoms_batch * 3) ** 2
        assert (
            n_entries_batch == pred_block.numel()
        ), f"Number of entries in batch {b} does not match"
        true_block = true_hessian[past_entries : past_entries + n_entries_batch]
        # Compare the blocks
        pred_block = pred_block.reshape(true_block.shape)
        loss += loss_fn(pred_block, true_block)
        atom_offset += n_atoms_batch
        past_entries += n_entries_batch
    # invert the mask
    test_mask = 1 - test_mask
    # all other (non-visited) entries should be zero
    assert torch.allclose(pred_hessian * test_mask, torch.zeros_like(pred_hessian))
    return loss


if __name__ == "__main__":

    project_root = os.path.dirname(os.path.dirname(__file__))

    for _hessian_build_method in ["1d", "blockdiagonal"]:
        config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        model_config = config["model"]
        model_config["do_hessian"] = True
        model_config["otf_graph"] = False
        model_config["hessian_build_method"] = _hessian_build_method
        model = EquiformerV2_OC20(**model_config)

        checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
        state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
        state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        model.train()
        model.to("cuda")

        from ocpmodels.ff_lmdb import LmdbDataset

        dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
        transform = HessianGraphTransform(
            cutoff=model.cutoff,
            max_neighbors=model.max_neighbors,
            use_pbc=model.use_pbc,
        )
        dataset = LmdbDataset(dataset_path, transform=transform)

        follow_batch = ["diag_ij", "edge_index", "message_idx_ij"]

        print("\n" + "=" * 100)
        print("\n# Single sample")
        dataloader = TGDataLoader(
            dataset, batch_size=1, shuffle=False, follow_batch=follow_batch
        )
        for _b, batch_base in tqdm(
            enumerate(dataloader), desc="Evaluating", total=len(dataloader)
        ):
            N = batch_base.natoms.sum().item()
            batch = batch_base.clone()
            batch = batch.to(model.device)
            batch = compute_extra_props(batch, pos_require_grad=False)
            energy, forces, out = model.forward(
                batch, eigen=True, hessian=True, return_l_features=True
            )
            pred_hessian = out["hessian"]

            print("\n" + "-")
            print(
                f"## Checking each method separately for B={batch.batch.max().item() + 1}"
            )
            run_hessian_tests(
                batch.edge_index,
                out["l012_edge_features"],
                out["l012_node_features"],
                batch,
            )

            if _b == 1:
                print("\n" + "-")
                equivariance_test(model, batch)
                print("symmetry:")
                diff = pred_hessian - pred_hessian.T
                print(f"Symmetry abs diff: {diff.abs().mean().item():.2e}")
                print(
                    f"Symmetry rel diff: {(diff.abs() / pred_hessian.abs()).mean().item():.2e}"
                )
                break

        print("\n" + "=" * 100)
        print("\n# Batching")
        dataloader = TGDataLoader(
            dataset, batch_size=2, shuffle=False, follow_batch=follow_batch
        )
        for batch_base in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            N = batch_base.natoms.sum().item()

            print("\n" + "-")
            print("## Checking gradients")

            batch = batch_base.clone()
            batch = batch.to(model.device)
            batch = compute_extra_props(batch, pos_require_grad=False)
            energy, forces, out = model.forward(
                batch, eigen=True, hessian=True, return_l_features=True
            )
            pred_hessian = out["hessian"]

            true_hessian = batch.hessian
            # true_hessian = true_hessian.reshape(pred_hessian.shape)

            # compute loss
            loss_fn = torch.nn.functional.mse_loss
            if pred_hessian.numel() > true_hessian.numel():
                # we computed a block diagonal hessian
                loss = compute_loss_blockdiagonal_hessian(
                    pred_hessian, true_hessian, loss_fn, batch
                )
            else:
                loss = loss_fn(pred_hessian, true_hessian)

            # backprop
            loss.backward()

            grad = []
            none_grad = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        grad.append(param.grad.norm().item())
                    else:
                        none_grad += 1
            print(f"num grad entries: {len(grad)} (none: {none_grad})")
            grad = torch.tensor(grad)
            print(f"Grad norm: {grad.mean().item():.2e}")

            print("\n" + "-")
            print(
                f"## Checking each method separately for B={batch.batch.max().item() + 1}"
            )

            run_hessian_tests(
                batch.edge_index,
                out["l012_edge_features"],
                out["l012_node_features"],
                batch,
            )

            break

    print("\n\nPassed! ✅")
