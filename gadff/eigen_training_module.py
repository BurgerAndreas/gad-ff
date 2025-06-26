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


class EigenPotentialModule(PotentialModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        # Freeze all parameters except the specified heads
        self.heads_to_train = [
            "eigval_1_head",
            "eigval_2_head",
            "eigvec_1_head",
            "eigvec_2_head",
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

        # Eigenvectors of real symmetric matrices are only unique up to a sign
        # So we need sign invariant loss functions
        # Store as private attributes to avoid PyTorch module registration
        self._loss_fn_vec = get_vector_loss_fn(training_config["loss_type_vec"])
        self._loss_fn = get_scalar_loss_fn(training_config["loss_type"])

        # self.loss_fn = nn.L1Loss()
        # self.MAEEval = MeanAbsoluteError()
        # self.MAPEEval = MeanAbsolutePercentageError()
        # self.cosineEval = CosineSimilarity(reduction="mean")

    @property
    def loss_fn(self):
        """Access the scalar loss function."""
        return self._loss_fn

    @property
    def loss_fn_vec(self):
        """Access the vector loss function."""
        return self._loss_fn_vec

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

    def load_model_state_dict(
        self, checkpoint: Mapping[str, Any], strict: bool = True
    ) -> None:
        print(f"EigenPotentialModule: Loading model state dict with strict: {strict}")
        print(f"Checkpoint keys: {checkpoint.keys()}")
        print(f"Checkpoint state_dict keys: {checkpoint['state_dict'].keys()}")
        # assert self.lightning_module is not None
        # self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)
        super().load_model_state_dict(checkpoint, strict)

    @torch.enable_grad()
    def compute_loss(self, batch):
        batch.pos.requires_grad_()
        batch = compute_extra_props(batch=batch, pos_require_grad=self.pos_require_grad)

        hat_ae, hat_forces, outputs = self.potential.forward(
            batch.to(self.device), eigen=True
        )

        loss_eigval1 = torch.tensor(0.0)
        loss_eigval2 = torch.tensor(0.0)
        loss_eigvec1 = torch.tensor(0.0)
        loss_eigvec2 = torch.tensor(0.0)

        if self.model_config["do_eigval_1"]:
            eigval_1 = outputs["eigval_1"]
            loss_eigval1 = self.loss_fn(eigval_1, batch.hessian_eigenvalue_1)
        if self.model_config["do_eigval_2"]:
            eigval_2 = outputs["eigval_2"]
            loss_eigval2 = self.loss_fn(eigval_2, batch.hessian_eigenvalue_2)
        if self.model_config["do_eigvec_1"]:
            eigvec_1 = outputs["eigvec_1"]
            loss_eigvec1 = self.loss_fn_vec(eigvec_1, batch.hessian_eigenvector_1)
        if self.model_config["do_eigvec_2"]:
            eigvec_2 = outputs["eigvec_2"]
            loss_eigvec2 = self.loss_fn_vec(eigvec_2, batch.hessian_eigenvector_2)

        info = {
            "Loss eigval1": loss_eigval1.detach().item(),
            "Loss eigval2": loss_eigval2.detach().item(),
            "Loss eigvec1": loss_eigvec1.detach().item(),
            "Loss eigvec2": loss_eigvec2.detach().item(),
        }
        self.MAEEval.reset()
        self.MAPEEval.reset()
        self.cosineEval.reset()

        loss = (
            (self.training_config["weight_eigval1"] * loss_eigval1)
            + (self.training_config["weight_eigval2"] * loss_eigval2)
            + (self.training_config["weight_eigvec1"] * loss_eigvec1)
            + (self.training_config["weight_eigvec2"] * loss_eigvec2)
        )
        return loss, info

    def compute_eval_loss(self, batch):
        """Compute comprehensive evaluation metrics for eigenvalues and eigenvectors."""
        batch = compute_extra_props(batch=batch, pos_require_grad=self.pos_require_grad)

        hat_ae, hat_forces, outputs = self.potential.forward(
            batch.to(self.device), eigen=True
        )

        eigval_1_pred = None
        eigval_2_pred = None
        eigvec_1_pred = None
        eigvec_2_pred = None

        if self.model_config["do_eigval_1"]:
            eigval_1_pred = outputs["eigval_1"]
        if self.model_config["do_eigval_2"]:
            eigval_2_pred = outputs["eigval_2"]
        if self.model_config["do_eigvec_1"]:
            eigvec_1_pred = outputs["eigvec_1"]
        if self.model_config["do_eigvec_2"]:
            eigvec_2_pred = outputs["eigvec_2"]

        eigval_1_true = batch.hessian_eigenvalue_1
        eigval_2_true = batch.hessian_eigenvalue_2
        eigvec_1_true = batch.hessian_eigenvector_1
        eigvec_2_true = batch.hessian_eigenvector_2

        eval_metrics = {}

        ###################################################
        # Eigenvalue metrics
        if eigval_1_pred is not None:
            eval_metrics["RMSE eigval1"] = torch.sqrt(
                torch.mean((eigval_1_pred - eigval_1_true) ** 2)
            ).item()
        if eigval_2_pred is not None:
            eval_metrics["RMSE eigval2"] = torch.sqrt(
                torch.mean((eigval_2_pred - eigval_2_true) ** 2)
            ).item()

        # MAPE for eigenvalues (avoid division by zero)
        eps = 1e-8
        if eigval_1_pred is not None:
            eval_metrics["MAPE eigval1"] = (
                torch.mean(
                    torch.abs(
                        (eigval_1_pred - eigval_1_true)
                        / (torch.abs(eigval_1_true) + eps)
                    )
                ).item()
                * 100
            )
        if eigval_2_pred is not None:
            eval_metrics["MAPE eigval2"] = (
                torch.mean(
                    torch.abs(
                        (eigval_2_pred - eigval_2_true)
                        / (torch.abs(eigval_2_true) + eps)
                    )
                ).item()
                * 100
            )

        # Relative error for eigenvalues (mean over batch)
        if eigval_1_pred is not None:
            rel_err_eigval1 = torch.abs(eigval_1_pred - eigval_1_true) / (
                torch.abs(eigval_1_true) + eps
            )
            eval_metrics["RelErr eigval1"] = torch.mean(rel_err_eigval1).item()
        if eigval_2_pred is not None:
            rel_err_eigval2 = torch.abs(eigval_2_pred - eigval_2_true) / (
                torch.abs(eigval_2_true) + eps
            )
            eval_metrics["RelErr eigval2"] = torch.mean(rel_err_eigval2).item()

        # Sign agreement for eigenvalues (important for Hessian analysis)
        def sign_agreement(y_pred, y_true):
            pred_signs = torch.sign(y_pred)
            true_signs = torch.sign(y_true)
            agreement = (pred_signs == true_signs).float()
            return torch.mean(agreement).item()

        if eigval_1_pred is not None:
            eval_metrics["SignCorrect eigval1"] = sign_agreement(
                eigval_1_pred, eigval_1_true
            )
        if eigval_2_pred is not None:
            eval_metrics["SignCorrect eigval2"] = sign_agreement(
                eigval_2_pred, eigval_2_true
            )

        # Both signs are correct simultaneously
        if eigval_1_pred is not None and eigval_2_pred is not None:
            pred_signs_1 = torch.sign(eigval_1_pred)
            true_signs_1 = torch.sign(eigval_1_true)
            pred_signs_2 = torch.sign(eigval_2_pred)
            true_signs_2 = torch.sign(eigval_2_true)

            both_signs_correct = (
                (pred_signs_1 == true_signs_1) & (pred_signs_2 == true_signs_2)
            ).float()
            eval_metrics["BothSignsCorrect"] = torch.mean(both_signs_correct).item()

        # Index 1 saddle point classification metrics (one negative, one positive eigenvalue)
        def is_index1_saddle(eigval1, eigval2):
            """Check if eigenvalues represent index 1 saddle point (one neg, one pos)"""
            sign1 = torch.sign(eigval1)
            sign2 = torch.sign(eigval2)
            return ((sign1 < 0) & (sign2 > 0)) | ((sign1 > 0) & (sign2 < 0))

        if eigval_1_pred is not None and eigval_2_pred is not None:
            true_saddle1 = is_index1_saddle(eigval_1_true, eigval_2_true)
            eval_metrics["TrueSaddle1"] = true_saddle1.float().mean().item()
            pred_saddle1 = is_index1_saddle(eigval_1_pred, eigval_2_pred)

            # Classification metrics for index 1 saddle points
            tp_saddle1 = (true_saddle1 & pred_saddle1).float().sum()
            fp_saddle1 = (~true_saddle1 & pred_saddle1).float().sum()
            fn_saddle1 = (true_saddle1 & ~pred_saddle1).float().sum()
            tn_saddle1 = (~true_saddle1 & ~pred_saddle1).float().sum()

            total_samples = len(true_saddle1)

            eval_metrics["TP Saddle1 (up)"] = tp_saddle1.item()
            eval_metrics["FP Saddle1 (low)"] = fp_saddle1.item()
            eval_metrics["FN Saddle1 (up)"] = fn_saddle1.item()
            eval_metrics["TN Saddle1 (low)"] = tn_saddle1.item()

            # Derived metrics
            precision_saddle1 = tp_saddle1 / (tp_saddle1 + fp_saddle1 + eps)
            recall_saddle1 = tp_saddle1 / (tp_saddle1 + fn_saddle1 + eps)
            f1_saddle1 = (
                2
                * precision_saddle1
                * recall_saddle1
                / (precision_saddle1 + recall_saddle1 + eps)
            )
            accuracy_saddle1 = (tp_saddle1 + tn_saddle1) / total_samples

            eval_metrics["Precision Saddle1"] = precision_saddle1.item()
            eval_metrics["Recall Saddle1"] = recall_saddle1.item()
            eval_metrics["F1 Saddle1"] = f1_saddle1.item()
            eval_metrics["Accuracy Saddle1"] = accuracy_saddle1.item()

        # Eigenvector metrics
        # Cosine similarity (most important for vectors)

        if eigvec_1_pred is not None:
            eval_metrics["CosSim eigvec1"] = cosine_similarity(
                eigvec_1_pred, eigvec_1_true
            ).item()
        if eigvec_2_pred is not None:
            eval_metrics["CosSim eigvec2"] = cosine_similarity(
                eigvec_2_pred, eigvec_2_true
            ).item()

        # Angular error in degrees
        def angular_error(v1, v2):
            cos_sim = cosine_similarity(v1, v2)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Numerical stability
            angle_rad = torch.acos(
                torch.abs(cos_sim)
            )  # abs because eigenvectors can have opposite signs
            return torch.rad2deg(angle_rad)

        if eigvec_1_pred is not None:
            eval_metrics["AngleErr eigvec1"] = angular_error(
                eigvec_1_pred, eigvec_1_true
            ).item()
        if eigvec_2_pred is not None:
            eval_metrics["AngleErr eigvec2"] = angular_error(
                eigvec_2_pred, eigvec_2_true
            ).item()

        # Vector magnitude error
        if eigvec_1_pred is not None:
            mag_1_pred = torch.norm(eigvec_1_pred, dim=-1)
            mag_1_true = torch.norm(eigvec_1_true, dim=-1)
            eval_metrics["MAE eigvec1 magnitude"] = torch.mean(
                torch.abs(mag_1_pred - mag_1_true)
            ).item()

        if eigvec_2_pred is not None:
            mag_2_pred = torch.norm(eigvec_2_pred, dim=-1)
            mag_2_true = torch.norm(eigvec_2_true, dim=-1)
            eval_metrics["MAE eigvec2 magnitude"] = torch.mean(
                torch.abs(mag_2_pred - mag_2_true)
            ).item()

        return eval_metrics

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        # compute training loss on eval set
        loss, info = self.compute_loss(batch)
        detached_loss = loss.detach()
        info["totloss"] = detached_loss.item()

        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        del info

        # compute eval metrics on eval set
        eval_info = self.compute_eval_loss(batch)
        for k, v in eval_info.items():
            info_prefix[f"{prefix}-{k}"] = v

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_step_outputs.append(outputs)

    def on_validation_epoch_end(self):

        val_epoch_metrics = average_over_batch_metrics(self.val_step_outputs)

        # print all keys and values
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
            # pretty_print(self.current_epoch, val_epoch_metrics["totloss"], prefix="val")

        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

        self.val_step_outputs.clear()
