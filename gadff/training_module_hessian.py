"""
Pytorch Lightning training module for predicting the full Hessian matrix.
"""

from typing import Dict, List, Optional, Tuple, Any, Mapping
from collections.abc import Iterable
from omegaconf import ListConfig
import os
import yaml
from pathlib import Path
import wandb

import torch
from torch import nn
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    StepLR,
    ConstantLR,
)
from torch_geometric.loader import DataLoader as TGDataLoader
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    CosineSimilarity,
)
from torch_scatter import scatter_mean

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from gadff.horm.ff_lmdb import LmdbDataset
from ocpmodels.hessian_graph_transform import HessianGraphTransform
from gadff.horm.utils import average_over_batch_metrics, pretty_print
import gadff.horm.utils as diff_utils
from gadff.path_config import find_project_root
from gadff.horm.training_module import (
    PotentialModule,
    compute_extra_props,
    SchemaUniformDataset,
)
from gadff.loss_functions import (
    compute_loss_blockdiagonal_hessian,
    get_hessian_loss_fn,
    get_eigval_eigvec_metrics,
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

        # model_config["do_hessian"] = True
        # model_config["otf_graph"] = False
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

        self.loss_fn_hessian = nn.MSELoss()
        print(f"Training config: {training_config['eigen_loss']}")
        self.loss_fn_eigen = get_hessian_loss_fn(**training_config["eigen_loss"])

        _alpha = self.training_config["eigen_loss"]["alpha"]
        if isinstance(_alpha, Iterable) or (isinstance(_alpha, float) and _alpha > 0.0):
            self.do_eigen_loss = True
            print("! Training with eigenvalue loss")
        else:
            self.do_eigen_loss = False
            print("! Training without eigenvalue loss")

        # loss from Hamiltonian prediction paper
        self.test_loss_fn_wa2 = get_hessian_loss_fn(
            loss_name="wa",
            k=2,
            alpha=1.0,
        )
        self.test_loss_fn_wa8 = get_hessian_loss_fn(
            loss_name="wa",
            k=8,
            alpha=1.0,
        )
        # Luca's loss
        self.test_loss_fn_eigen = get_hessian_loss_fn(
            loss_name="eigenspectrum", k=None, alpha=1.0
        )
        self.test_loss_fn_eigen_k2 = get_hessian_loss_fn(
            loss_name="eigenspectrum", k=2, alpha=1.0
        )
        self.test_loss_fn_eigen_k8 = get_hessian_loss_fn(
            loss_name="eigenspectrum", k=8, alpha=1.0
        )

        # self.loss_fn = nn.L1Loss()
        # self.MAEEval = MeanAbsoluteError()
        # self.MAPEEval = MeanAbsolutePercentageError()
        # self.cosineEval = CosineSimilarity(reduction="mean")

        # # Store functions as private attributes to avoid PyTorch module registration
        # self._loss_fn_eigen = get_vector_loss_fn(training_config["loss_type_vec"])
        # self._loss_fn_hessian = get_scalar_loss_fn(training_config["loss_type"])

    # @property
    # def loss_fn_hessian(self):
    #     """Access the scalar loss function."""
    #     return self._loss_fn_hessian

    # @property
    # def loss_fn_eigen(self):
    #     """Access the vector loss function."""
    #     return self._loss_fn_eigen

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

    def setup(self, stage: Optional[str] = None):
        # Add SLURM job ID to config if it exists in environment
        if "SLURM_JOB_ID" in os.environ:
            slurm_job_id = os.environ["SLURM_JOB_ID"]
            try:
                wandb.log({"slurm_job_id": slurm_job_id}, step=self.global_step)
            except Exception as e:
                print(f"Error logging SLURM job ID: {e}")
            print(f"SLURM job ID: {slurm_job_id}")
        print("Setting up dataset")
        if stage == "fit":
            print(f"Loading training dataset from {self.training_config['trn_path']}")
            if (
                isinstance(self.training_config["trn_path"], list)
                or isinstance(self.training_config["trn_path"], tuple)
                or isinstance(self.training_config["trn_path"], ListConfig)
            ):
                datasets = []
                for path in self.training_config["trn_path"]:
                    transform = HessianGraphTransform(
                        cutoff=self.potential.cutoff,
                        max_neighbors=self.potential.max_neighbors,
                        use_pbc=self.potential.use_pbc,
                    )
                    dataset = LmdbDataset(
                        Path(path),
                        transform=transform,
                        **self.training_config,
                    )
                    datasets.append(SchemaUniformDataset(dataset))
                    print(f"Loaded dataset from {path} with {len(dataset)} samples")

                # Combine all datasets into a single concatenated dataset
                self.train_dataset = ConcatDataset(datasets)
                print(
                    f"Combined {len(datasets)} datasets into one with {len(self.train_dataset)} total samples"
                )
            else:
                transform = HessianGraphTransform(
                    cutoff=self.potential.cutoff,
                    max_neighbors=self.potential.max_neighbors,
                    use_pbc=self.potential.use_pbc,
                )
                self.train_dataset = LmdbDataset(
                    Path(self.training_config["trn_path"]),
                    transform=transform,
                    **self.training_config,
                )
            transform = HessianGraphTransform(
                cutoff=self.potential.cutoff,
                max_neighbors=self.potential.max_neighbors,
                use_pbc=self.potential.use_pbc,
            )
            self.val_dataset = LmdbDataset(
                Path(self.training_config["val_path"]),
                transform=transform,
                **self.training_config,
            )
            print("# of training data: ", len(self.train_dataset))
            print("# of validation data: ", len(self.val_dataset))

        else:
            raise NotImplementedError
        return

    @torch.enable_grad()
    def compute_loss(self, batch):
        batch.pos.requires_grad_()
        batch = compute_extra_props(batch, pos_require_grad=self.pos_require_grad)

        hat_ae, hat_forces, outputs = self.potential.forward(
            batch.to(self.device), hessian=True
        )
        hat_hessian = outputs["hessian"].to(self.device)
        hessian_true = batch.hessian.to(self.device)
        hessian_loss = self.loss_fn_hessian(hat_hessian, hessian_true)
        loss = hessian_loss * self.training_config["hessian_loss_weight"]
        info = {
            "Loss Hessian": hessian_loss.detach().item(),
        }

        if self.do_eigen_loss:
            eigen_loss = self.loss_fn_eigen(
                pred=hat_hessian,
                target=hessian_true,
                data=batch,
            )
            loss += eigen_loss
            info["Loss Eigen"] = eigen_loss.detach().item()

        # self.MAEEval.reset()
        # self.MAPEEval.reset()
        # self.cosineEval.reset()

        return loss, info

    def compute_eval_loss(self, batch, prefix):
        """Compute comprehensive evaluation metrics for eigenvalues and eigenvectors."""
        batch = compute_extra_props(batch=batch, pos_require_grad=self.pos_require_grad)

        hat_ae, hat_forces, outputs = self.potential.forward(
            batch.to(self.device), hessian=True
        )

        hessian_true = batch.hessian
        hessian_pred = outputs["hessian"]

        eval_metrics = {}
        eval_metrics["Loss Eigen"] = (
            self.test_loss_fn_eigen(
                pred=hessian_pred,
                target=hessian_true,
                data=batch,
                debugstr=f"{prefix}-step{self.global_step}-epoch{self.current_epoch}-Loss Eigen",
            )
            .detach()
            .item()
        )
        eval_metrics["Loss Eigen k2"] = (
            self.test_loss_fn_eigen_k2(
                pred=hessian_pred,
                target=hessian_true,
                data=batch,
                debugstr=f"{prefix}-step{self.global_step}-epoch{self.current_epoch}-Loss Eigen k2",
            )
            .detach()
            .item()
        )
        eval_metrics["Loss Eigen k8"] = (
            self.test_loss_fn_eigen_k8(
                pred=hessian_pred,
                target=hessian_true,
                data=batch,
                debugstr=f"{prefix}-step{self.global_step}-epoch{self.current_epoch}-Loss Eigen k8",
            )
            .detach()
            .item()
        )
        # loss from Hamiltonian prediction paper
        eval_metrics["Loss WA k2"] = (
            self.test_loss_fn_wa2(
                pred=hessian_pred,
                target=hessian_true,
                data=batch,
                debugstr=f"{prefix}-step{self.global_step}-epoch{self.current_epoch}-Loss WA k2",
            )
            .detach()
            .item()
        )
        eval_metrics["Loss WA k8"] = (
            self.test_loss_fn_wa8(
                pred=hessian_pred,
                target=hessian_true,
                data=batch,
                debugstr=f"{prefix}-step{self.global_step}-epoch{self.current_epoch}-Loss WA k8",
            )
            .detach()
            .item()
        )

        # Eigenvalue, Eigenvector metrics
        eig_metrics = get_eigval_eigvec_metrics(
            hessian_true,
            hessian_pred,
            batch,
            prefix=f"{prefix}-step{self.global_step}-epoch{self.current_epoch}",
        )
        eval_metrics.update(eig_metrics)

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
        eval_info = self.compute_eval_loss(batch, prefix=prefix)
        for k, v in eval_info.items():
            info_prefix[f"{prefix}-{k}"] = v

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)
