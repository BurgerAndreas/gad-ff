"""
Pytorch Lightning training module for predicting Hessian eigenvalues and eigenvectors.
"""

from typing import Dict, List, Optional, Tuple, Any, Mapping
import os
from pathlib import Path
import torch
from torch import nn

from torch_geometric.loader import DataLoader as TGDataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    StepLR,
    ConstantLR,
)
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    CosineSimilarity,
)
from torch_scatter import scatter_mean
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.horm.utils import average_over_batch_metrics, pretty_print
import gadff.horm.utils as diff_utils
import yaml
from gadff.path_config import find_project_root
from gadff.horm.training_module import PotentialModule, compute_extra_props
from gadff.loss_functions import (
    get_vector_loss_fn,
    get_scalar_loss_fn,
    cosine_similarity,
)


class MyPLTrainer(pl.Trainer):
    # Does not do anything?
    # self.trainer.strategy.load_model_state_dict
    def load_model_state_dict(
        self, checkpoint: Mapping[str, Any], strict: bool = True
    ) -> None:
        print(f"MyPLTrainer: Loading model state dict with strict: {strict}")
        print(f"Checkpoint keys: {checkpoint.keys()}")
        print(f"Checkpoint state_dict keys: {checkpoint['state_dict'].keys()}")
        # assert self.lightning_module is not None
        # self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)
        super().load_model_state_dict(checkpoint, strict)


class HessianPotentialModule(PotentialModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        # Freeze all parameters except the specified heads
        self.heads_to_train = [
            "hessian_layers",
            "hessian_head",
            "hessian_edge_message_proj",
            "hessian_node_proj",
        ]

        super().__init__(
            model_config=model_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
        )

        # For Lightning
        # Allow non-strict checkpoint loading for transfer learning
        self.strict_loading = False

        # Only needed to predict forces from energy of Hessian from forces
        self.pos_require_grad = False

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        # self.MAEEval = MeanAbsoluteError()
        # self.MAPEEval = MeanAbsolutePercentageError()
        # self.cosineEval = CosineSimilarity(reduction="mean")

    #     # Store functions as private attributes to avoid PyTorch module registration
    #     self._loss_fn_vec = get_vector_loss_fn(training_config["loss_type_vec"])
    #     self._loss_fn = get_scalar_loss_fn(training_config["loss_type"])

    # @property
    # def loss_fn(self):
    #     """Access the scalar loss function."""
    #     return self._loss_fn

    # @property
    # def loss_fn_vec(self):
    #     """Access the vector loss function."""
    #     return self._loss_fn_vec

    def _freeze_except_heads(self, heads_to_train: List[str]) -> None:
        """
        Freeze all model parameters except the specified heads.

        Args:
            heads_to_train: List of head names to keep trainable
        """
        # First, freeze all parameters
        for param in self.potential.parameters():
            param.requires_grad = False

        # Then unfreeze only the specified heads
        for head_name in heads_to_train:
            if hasattr(self.potential, head_name):
                head_module = getattr(self.potential, head_name)
                if head_module is not None:
                    for param in head_module.parameters():
                        param.requires_grad = True
                    print(f"Unfroze parameters for {head_name}")
                else:
                    print(
                        f"Warning: {head_name} is None - head not created during model initialization"
                    )
            else:
                print(f"Warning: {head_name} not found in model")

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.potential.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.potential.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)"
        )

    def configure_optimizers(self):
        print("Configuring optimizer")
        # Only optimize parameters that require gradients (unfrozen heads)
        if self.training_config["train_heads_only"]:
            self._freeze_except_heads(self.heads_to_train)
        trainable_params = [p for p in self.potential.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, **self.optimizer_config)

        if self.training_config["lr_schedule_type"] is not None:
            LR_SCHEDULER = {
                "cos": CosineAnnealingWarmRestarts,
                "step": StepLR,
                "constant": ConstantLR,
            }
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    @torch.enable_grad()
    def compute_loss(self, batch):
        batch.pos.requires_grad_()
        batch = compute_extra_props(batch, pos_require_grad=self.pos_require_grad)

        hat_ae, hat_forces, outputs = self.potential.forward(
            batch.to(self.device),
        )
        hat_hessian = outputs["hessian"].to(self.device)
        hessian_true = batch.hessian.to(self.device)
        hessian_loss = self.loss_fn(hat_hessian, hessian_true)
        info = {
            "MAE_hessian": hessian_loss.detach().item(),
        }
        self.MAEEval.reset()
        self.MAPEEval.reset()
        self.cosineEval.reset()

        loss = hessian_loss
        return loss, info

    # def compute_eval_loss(self, batch):
    #     """Compute comprehensive evaluation metrics for eigenvalues and eigenvectors."""
    #     batch = compute_extra_props(batch=batch, pos_require_grad=self.pos_require_grad)

    #     hat_ae, hat_forces, outputs = self.potential.forward(
    #         batch.to(self.device), eigen=True
    #     )

    #     hessian_true = batch.hessian
    #     hessian_pred = outputs["hessian"]

    #     eval_metrics = {}

    #     # TODO
    #     return eval_metrics

    # def _shared_eval(self, batch, batch_idx, prefix, *args):
    #     # compute training loss on eval set
    #     loss, info = self.compute_loss(batch)
    #     detached_loss = loss.detach()
    #     info["totloss"] = detached_loss.item()

    #     info_prefix = {}
    #     for k, v in info.items():
    #         info_prefix[f"{prefix}-{k}"] = v
    #     del info

    #     # compute eval metrics on eval set
    #     eval_info = self.compute_eval_loss(batch)
    #     for k, v in eval_info.items():
    #         info_prefix[f"{prefix}-{k}"] = v

    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     return info_prefix
