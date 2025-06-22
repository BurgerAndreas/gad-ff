"""
Pytorch Lightning training module for predicting Hessian eigenvalues and eigenvectors.
"""
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import torch
from torch import nn

from torch_geometric.loader import DataLoader as TGDataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    StepLR,
    # CosineAnnealingLR,
)
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

class EigenPotentialModule(PotentialModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        # Freeze all parameters except the specified heads
        self.heads_to_train = ['eigval_1_head', 'eigval_2_head', 'eigvec_1_head', 'eigvec_2_head']
            
        super().__init__(
            model_config=model_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
        )
        
        # Only needed to predict forces from energy of Hessian from forces
        self.pos_require_grad = False
        
        if training_config["loss_type"] == "l1":
            self.loss_fn = nn.L1Loss()
        elif training_config["loss_type"] == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss type: {training_config['loss_type']}")
        
        # self.loss_fn = nn.L1Loss()
        # self.MAEEval = MeanAbsoluteError()
        # self.MAPEEval = MeanAbsolutePercentageError()
        # self.cosineEval = CosineSimilarity(reduction="mean")
    
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
                    print(f"Warning: {head_name} is None - head not created during model initialization")
            else:
                print(f"Warning: {head_name} not found in model")
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.potential.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.potential.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def configure_optimizers(self):
        # Only optimize parameters that require gradients (unfrozen heads)
        self._freeze_except_heads(self.heads_to_train)
        trainable_params = [p for p in self.potential.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, **self.optimizer_config)

        if not self.training_config["lr_schedule_type"] is None:
            from torch.optim.lr_scheduler import (
                CosineAnnealingWarmRestarts,
                StepLR,
            )
            LR_SCHEDULER = {
                "cos": CosineAnnealingWarmRestarts,
                "step": StepLR,
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
        batch = compute_extra_props(
            batch=batch, 
            pos_require_grad=self.pos_require_grad
        )
        
        hat_ae, hat_forces, outputs = self.potential.forward(
            batch.to(self.device), eigen=True
        )
        
        eigval_1 = outputs["eigval_1"]
        eigval_2 = outputs["eigval_2"]
        eigvec_1 = outputs["eigvec_1"]
        eigvec_2 = outputs["eigvec_2"]
        
        loss_eigval1 = self.loss_fn(eigval_1, batch.eigval_1)
        loss_eigval2 = self.loss_fn(eigval_2, batch.eigval_2)
        loss_eigvec1 = self.loss_fn(eigvec_1, batch.eigvec_1)
        loss_eigvec2 = self.loss_fn(eigvec_2, batch.eigvec_2)
        
        info = {
            "MAE_eigval1": loss_eigval1.detach().item(),
            "MAE_eigval2": loss_eigval2.detach().item(),
            "MAE_eigvec1": loss_eigvec1.detach().item(),
            "MAE_eigvec2": loss_eigvec2.detach().item(),
        }
        self.MAEEval.reset()
        self.MAPEEval.reset()
        self.cosineEval.reset()

        loss = (self.training_config["weight_eigval1"] * loss_eigval1) + \
            (self.training_config["weight_eigval2"] * loss_eigval2) + \
            (self.training_config["weight_eigvec1"] * loss_eigvec1) + \
            (self.training_config["weight_eigvec2"] * loss_eigvec2)
        return loss, info
        