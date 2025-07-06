"""
PyTorch Lightning module for training AlphaNet
"""

from typing import Dict, List, Optional, Tuple
from omegaconf import ListConfig
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import ConcatDataset

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
from gadff.path_config import find_project_root, fix_dataset_path


LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}
GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8])


class SchemaUniformDataset:
    """Wrapper that ensures all datasets have the same attributes.

    RGD1 lacks:
    ae: <class 'torch.Tensor'> torch.Size([]) torch.float32 -> same as energy
    rxn: <class 'torch.Tensor'> torch.Size([]) torch.int64 -> add -1 to all

    All other (T1x based) datasets lack:
    freq: <class 'torch.Tensor'> torch.Size([N*3])
    eig_values: <class 'torch.Tensor'> torch.Size([N*3])
    force_constant: <class 'torch.Tensor'> torch.Size([N*3])
    -> remove these attributes from the dataset
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Add missing attributes
        if not hasattr(data, "ae"):
            data.ae = torch.tensor(data.energy.item(), dtype=data.energy.dtype)
        if not hasattr(data, "rxn"):
            data.rxn = torch.tensor(-1, dtype=torch.int64)

        # Remove extra attributes
        if hasattr(data, "freq"):
            delattr(data, "freq")
        if hasattr(data, "eig_values"):
            delattr(data, "eig_values")
        if hasattr(data, "force_constant"):
            delattr(data, "force_constant")
        return data


def compute_extra_props(batch, pos_require_grad=True):
    """Adds device, z, and removes mean batch"""
    device = batch.energy.device
    indices = batch.one_hot.long().argmax(dim=1)
    batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    # atomization energy. shape used by equiformerv2
    if not hasattr(batch, "ae"):
        batch.ae = torch.zeros_like(batch.energy)
    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


class AlphaConfig:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)


class PotentialModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        super().__init__()

        self.model_config = model_config

        if self.model_config["name"] == "EquiformerV2":
            root_dir = find_project_root()
            config_path = os.path.join(root_dir, "configs/equiformer_v2.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(root_dir, "equiformer_v2.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            model_config = config["model"]
            model_config.update(self.model_config)
            self.potential = EquiformerV2_OC20(**model_config)
        elif self.model_config["name"] == "AlphaNet":
            from alphanet.models.alphanet import AlphaNet

            self.potential = AlphaNet(AlphaConfig(model_config)).float()
        elif (
            self.model_config["name"] == "LEFTNet"
            or self.model_config["name"] == "LEFTNet-df"
        ):
            from leftnet.potential import Potential
            from leftnet.model import LEFTNet

            leftnet_config = dict(
                pos_require_grad=True,
                cutoff=10.0,
                num_layers=6,
                hidden_channels=196,
                num_radial=96,
                in_hidden_channels=8,
                reflect_equiv=True,
                legacy=True,
                update=True,
                pos_grad=False,
                single_layer_output=True,
            )
            node_nfs: List[int] = [9] * 1  # 3 (pos) + 5 (cat) + 1 (charge)
            edge_nf: int = 0  # edge type
            condition_nf: int = 1
            fragment_names: List[str] = ["structure"]
            pos_dim: int = 3
            edge_cutoff: Optional[float] = None
            self.potential = Potential(
                model_config=leftnet_config,
                node_nfs=node_nfs,  # 3 (pos) + 5 (cat) + 1 (charge),
                edge_nf=edge_nf,
                condition_nf=condition_nf,
                fragment_names=fragment_names,
                pos_dim=pos_dim,
                edge_cutoff=edge_cutoff,
                model=LEFTNet,
                enforce_same_encoding=None,
                source=None,
                timesteps=5000,
                condition_time=False,
            )
        else:
            print(
                "Please Check your model name (choose from 'EquiformerV2', 'AlphaNet', 'LEFTNet', 'LEFTNet-df')"
            )
        training_config = self.fix_paths(training_config)
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.pos_require_grad = True

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = diff_utils.Queue()
            self.gradnorm_queue.add(3000)
        self.save_hyperparameters()

        self.loss_fn = nn.L1Loss()
        self.MAEEval = MeanAbsoluteError()
        self.MAPEEval = MeanAbsolutePercentageError()
        self.cosineEval = CosineSimilarity(reduction="mean")
        self.val_step_outputs = []

    def fix_paths(self, training_config):
        """
        Fix paths in the training config to be relative to the project root.
        """
        try:
            training_config["trn_path"] = fix_dataset_path(training_config["trn_path"])
            training_config["val_path"] = fix_dataset_path(training_config["val_path"])
        except Exception as e:
            pass
        return training_config

    def configure_optimizers(self):
        print("Configuring optimizer")
        optimizer = torch.optim.AdamW(
            self.potential.parameters(), **self.optimizer_config
        )

        if self.training_config["lr_schedule_type"] is not None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
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
                    dataset = LmdbDataset(
                        Path(path),
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
                self.train_dataset = LmdbDataset(
                    Path(self.training_config["trn_path"]),
                    **self.training_config,
                )
            self.val_dataset = LmdbDataset(
                Path(self.training_config["val_path"]),
                **self.training_config,
            )
            print("# of training data: ", len(self.train_dataset))
            print("# of validation data: ", len(self.val_dataset))

        else:
            raise NotImplementedError

    def get_jacobian(self, forces, pos, grad_outputs, create_graph=False, looped=False):
        """
        Compute the Jacobian of forces with respect to atomic positions (∂F/∂r).

        This function computes the negative Hessian matrix by taking derivatives of predicted
        forces with respect to atomic positions. The Jacobian is mathematically equivalent to
        the negative second derivative of energy: ∂F/∂r = -∂²E/∂r².

        Args:
            forces (torch.Tensor): Predicted forces tensor of shape (N_atoms, 3)
            pos (torch.Tensor): Atomic positions tensor of shape (N_atoms, 3) with requires_grad=True
            grad_outputs (torch.Tensor): Gradient output tensor specifying which Jacobian elements
                to compute. Shape varies: (num_samples, N_atoms, 3) or (num_samples, 3, N_atoms, 3)
            create_graph (bool, optional): Whether to create computation graph for higher-order
                derivatives. Defaults to False.
            looped (bool, optional): Whether to use explicit loops instead of vectorized vmap.
                Use when vmap causes memory issues. Defaults to False.

        Returns:
            torch.Tensor: Jacobian tensor containing ∂F_i/∂r_j for sampled indices.
                Shape matches grad_outputs dimensions.

        Note:
            - Uses torch.vmap for efficient vectorized computation when possible
            - Falls back to explicit loops when looped=True for memory efficiency
            - The grad_outputs tensor acts as a mask, computing only requested Jacobian elements
            - Essential for stochastic row sampling in Hessian-informed training
        """

        def compute_grad(grad_output):
            return torch.autograd.grad(
                outputs=forces,
                inputs=pos,
                grad_outputs=grad_output,
                create_graph=create_graph,
                retain_graph=True,
            )[0]

        if not looped:
            if len(grad_outputs.shape) == 4:
                compute_jacobian = torch.vmap(torch.vmap(compute_grad))
            else:
                compute_jacobian = torch.vmap(compute_grad)
            return compute_jacobian(grad_outputs)
        else:
            num_atoms = forces.shape[0]
            if len(grad_outputs.shape) == 4:
                full_jac = torch.zeros(grad_outputs.shape[0], 3, num_atoms, 3).to(
                    forces.device
                )
                for i in range(grad_outputs.shape[0]):
                    for j in range(3):
                        full_jac[i, j] = compute_grad(grad_outputs[i, j])
            else:
                full_jac = torch.zeros(grad_outputs.shape[0], num_atoms, 3).to(
                    forces.device
                )
                for i in range(grad_outputs.shape[0]):
                    full_jac[i] = compute_grad(grad_outputs[i])
            return full_jac

    def get_force_jac_loss(
        self,
        forces,
        batch,
        hessian_label,
        num_samples=2,
        looped=False,
        finite_differences=False,
        forward=None,
        collater=None,
    ):
        """
        Compute Hessian loss using stochastic row sampling strategy for efficient training.

        This function implements the core innovation of HORM's Hessian-informed training by using
        stochastic sampling to make second-order derivative training computationally tractable.
        Instead of computing expensive full Hessian matrices (N×3 × N×3), it randomly samples
        a small number of matrix elements per molecule and compares predicted vs. ground truth values.

        Args:
            forces (torch.Tensor): Predicted forces from the model, shape (total_atoms, 3)
            batch: Batch object containing molecular data including:
                - natoms (torch.Tensor): Number of atoms per molecule
                - pos (torch.Tensor): Atomic positions with requires_grad=True
                - hessian (torch.Tensor): Ground truth Hessian data from HORM dataset
            hessian_label (torch.Tensor): Flattened ground truth Hessian matrices
            num_samples (int, optional): Number of Hessian elements to sample per molecule.
                Defaults to 2.
            looped (bool, optional): Whether to use looped Jacobian computation. Defaults to False.
            finite_differences (bool, optional): Whether to use finite differences (unused).
                Defaults to False.
            forward (callable, optional): Forward function for finite differences (unused).
                Defaults to None.
            collater (callable, optional): Data collation function (unused). Defaults to None.

        Returns:
            torch.Tensor: Scalar Hessian loss value normalized by batch size and scaled by 1/10

        Implementation Details:
            1. **Stochastic Sampling**: Uses sample_with_mask() to randomly select matrix elements
            2. **Batch Processing**: Handles variable molecule sizes via cumulative indexing
            3. **Jacobian Computation**: Calls get_jacobian() with sparse grad_outputs
            4. **Loss Calculation**: L2 norm between predicted and true Jacobian elements
            5. **Outlier Filtering**: Scales loss for Hessians with extreme values (>10000)

        Note:
            This approach enables training with second-order information while maintaining
            computational efficiency, addressing the "dramatically increased cost and complexity"
            mentioned in the HORM paper. The stochastic sampling allows models to learn
            Hessian patterns without computing full expensive matrices.
        """

        natoms = batch.natoms
        total_num_atoms = forces.shape[0]

        mask = torch.ones(total_num_atoms, dtype=torch.bool)
        cumulative_sums = [0] + torch.cumsum(natoms, 0).tolist()

        by_molecule = []
        grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device)
        for i, atoms_in_mol in enumerate(batch.natoms):
            submask = mask[cumulative_sums[i] : cumulative_sums[i + 1]]
            samples = self.sample_with_mask(atoms_in_mol, num_samples, submask)

            by_molecule.append(samples)  # swap below and above line, crucial
            offset_samples = (
                samples.clone()
            )  # Create a copy of the samples array to avoid modifying the original
            offset_samples[:, 0] += cumulative_sums[i]
            # Vectorized assignment to grad_outputs
            grad_outputs[
                torch.arange(samples.shape[0]),
                offset_samples[:, 0],
                offset_samples[:, 1],
            ] = 1
        # Compute the jacobian using grad_outputs

        jac = self.get_jacobian(
            forces, batch.pos, grad_outputs, create_graph=True, looped=looped
        )
        # jac = self.get_jacobian_finite_difference(forces, batch, grad_outputs = grad_outputs, forward=self._forward)

        # Decomposing the Jacobian tensor by molecule in a batch
        mask_per_mol = [
            mask[cum_sum : cum_sum + nat]
            for cum_sum, nat in zip(cumulative_sums[:-1], natoms)
        ]
        num_free_atoms_per_mol = torch.tensor(
            [sum(sub_mask) for sub_mask in mask_per_mol], device=natoms.device
        )
        cum_jac_indexes = [0] + torch.cumsum(
            (num_free_atoms_per_mol * natoms) * 9, dim=0
        ).tolist()

        jacs_per_mol = [
            jac[: len(mol_samps), cum_sum : cum_sum + nat, :]
            for mol_samps, cum_sum, nat in zip(
                by_molecule, cumulative_sums[:-1], natoms
            )
        ]
        jacs_per_mol = [
            mol_jac[:, mask, :] for mol_jac, mask in zip(jacs_per_mol, mask_per_mol)
        ]  # do the same for te student hessians

        if torch.any(torch.isnan(jac)):
            raise Exception("FORCE JAC IS NAN")

        batch.fixed = torch.zeros(total_num_atoms)

        true_jacs_per_mol = []
        for i, samples in enumerate(by_molecule):
            fixed_atoms = batch.fixed[cumulative_sums[i] : cumulative_sums[i + 1]]
            fixed_cumsum = torch.cumsum(fixed_atoms, dim=0)
            num_free_atoms = num_free_atoms_per_mol[i]
            curr = hessian_label[cum_jac_indexes[i] : cum_jac_indexes[i + 1]].reshape(
                num_free_atoms, 3, natoms[i], 3
            )
            curr = curr[:, :, mask_per_mol[i], :]  # filter out the masked columns
            subsampled_curr = curr[
                (samples[:, 0] - fixed_cumsum[samples[:, 0]]).long(), samples[:, 1]
            ]  # get the sampled rows
            true_jacs_per_mol.append(subsampled_curr)

        # just copying what DDPLoss does for our special case
        def custom_loss(jac, true_jac):
            return torch.norm(jac - true_jac, p=2, dim=-1).sum(dim=1).mean(dim=0)

        losses = [
            custom_loss(-jac, true_jac)
            for jac, true_jac in zip(jacs_per_mol, true_jacs_per_mol)
        ]
        valid_losses = [
            loss * 1e-8 if true_jac.abs().max().item() > 10000 else loss
            for loss, true_jac in zip(losses, true_jacs_per_mol)
        ]  # filter weird hessians

        loss = sum(valid_losses)

        num_samples = batch.batch.max() + 1
        # Multiply by world size since gradients are averaged
        # across DDP replicas
        loss = loss / num_samples / 10
        return loss

    def sample_with_mask(self, n, num_samples, mask):
        if mask.shape[0] != n:
            raise ValueError(
                "Mask length must be equal to the number of rows in the grid (n)"
            )

        # Calculate total available columns after applying the mask
        # Only rows where mask is True are considered
        valid_rows = torch.where(mask)[0]  # Get indices of rows that are True
        if valid_rows.numel() == 0:
            raise ValueError("No valid rows available according to the mask")

        # Each valid row contributes 3 indices
        valid_indices = valid_rows.repeat_interleave(3) * 3 + torch.tensor(
            [0, 1, 2]
        ).repeat(valid_rows.size(0)).to(mask.device)

        # Sample unique indices from the valid indices
        chosen_indices = valid_indices[
            torch.randperm(valid_indices.size(0))[:num_samples]
        ]

        # Convert flat indices back to row and column indices
        row_indices = chosen_indices // 3
        col_indices = chosen_indices % 3

        # Combine into 2-tuples
        samples = torch.stack((row_indices, col_indices), dim=1)

        return samples

    def train_dataloader(self) -> TGDataLoader:
        return TGDataLoader(
            self.train_dataset,
            batch_size=self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
        )

    def val_dataloader(self) -> TGDataLoader:
        return TGDataLoader(
            self.val_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    def test_dataloader(self) -> TGDataLoader:
        return TGDataLoader(
            self.test_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    @torch.enable_grad()
    def compute_loss(self, batch):
        batch.pos.requires_grad_()
        batch = compute_extra_props(batch, pos_require_grad=self.pos_require_grad)
        if self.model_config["name"] == "LEFTNet":
            hat_ae, hat_forces = self.potential.forward_autograd(
                batch.to(self.device),
            )
        else:
            hat_ae, hat_forces, out = self.potential.forward(
                batch.to(self.device),
            )
        hat_ae = hat_ae.squeeze().to(self.device)
        hat_forces = hat_forces.to(self.device)
        ae = batch.ae.to(self.device)
        forces = batch.forces.to(self.device)
        hessian_loss = self.get_force_jac_loss(hat_forces, batch, batch.hessian)
        eloss = self.loss_fn(ae, hat_ae)
        floss = self.loss_fn(forces, hat_forces)
        info = {
            "MAE_E": eloss.detach().item(),
            "MAE_F": floss.detach().item(),
            "MAE_hessian": hessian_loss.detach().item(),
        }
        self.MAEEval.reset()
        self.MAPEEval.reset()
        self.cosineEval.reset()

        loss = floss * 100 + eloss * 4 + hessian_loss * 4
        return loss, info

    def training_step(self, batch, batch_idx):
        loss, info = self.compute_loss(batch)

        self.log("train-totloss", loss, rank_zero_only=True)

        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)
        del info
        return loss

    def __shared_eval(self, batch, batch_idx, prefix, *args):
        with torch.enable_grad():
            loss, info = self.compute_loss(batch)
            info["totloss"] = loss.item()

            info_prefix = {}
            for k, v in info.items():
                key = f"{prefix}-{k}"
                if isinstance(v, torch.Tensor):
                    v = v.detach()
                    if v.is_cuda:
                        v = v.cpu()
                    if v.numel() == 1:
                        info_prefix[key] = v.item()
                    else:
                        info_prefix[key] = v.numpy()
                else:
                    info_prefix[key] = v
                self.log(
                    key, v, on_step=True, on_epoch=True, prog_bar=True, logger=True
                )

            del info
        return info_prefix

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        loss, info = self.compute_loss(batch)
        detached_loss = loss.detach()
        info["totloss"] = detached_loss.item()
        # info["totloss"] = loss.item()

        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        del info

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
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")

        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

        self.val_step_outputs.clear()

    def _configure_gradient_clipping(
        self,
        optimizer,
        # optimizer_idx,
        gradient_clip_val,
        gradient_clip_algorithm,
    ):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        max_grad_norm = 2 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = diff_utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
