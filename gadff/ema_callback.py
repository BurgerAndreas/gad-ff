"""
PyTorch Lightning EMA Callback using PyTorch's official AveragedModel.

This implementation uses torch.optim.swa_utils.AveragedModel for robust,
well-tested exponential moving average functionality.
"""

from typing import Any, Dict, Optional
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_info
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# alternative
# from ocpmodels.modules.exponential_moving_average import (
#     ExponentialMovingAverage,
# )


class EMACallback(Callback):
    """
    Exponential Moving Average callback using PyTorch's AveragedModel.
    
    This callback maintains exponential moving averages of model parameters
    using PyTorch's official implementation for optimal performance and reliability.
    
    Args:
        decay: EMA decay factor (0.999 is typical, higher = slower EMA updates)
        validate_with_ema: Whether to use EMA weights during validation
        save_ema_state: Whether to save EMA state in checkpoints
        use_buffers: Whether to apply EMA to model buffers (BatchNorm stats, etc.)
    """
    
    def __init__(
        self,
        decay: float = 0.999,
        validate_with_ema: bool = True,
        save_ema_state: bool = True,
        use_buffers: bool = True,
    ):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
            
        self.decay = decay
        self.validate_with_ema = validate_with_ema
        self.save_ema_state = save_ema_state
        self.use_buffers = use_buffers
        self.ema_model: Optional[AveragedModel] = None
        
        rank_zero_info(f"EMA Callback initialized with decay={decay}, use_buffers={use_buffers}")

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize EMA model when training starts."""
        if stage == "fit":
            # Create EMA model using PyTorch's AveragedModel
            self.ema_model = AveragedModel(
                model=pl_module,
                multi_avg_fn=get_ema_multi_avg_fn(self.decay),
                use_buffers=self.use_buffers,
            )
            rank_zero_info(f"EMA model initialized with decay={self.decay}")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after each training step."""
        if self.ema_model is not None:
            self.ema_model.update_parameters(pl_module)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Store current weights and apply EMA weights for validation."""
        if self.ema_model is not None and self.validate_with_ema:
            # Store original model parameters
            self._store_original_params(pl_module)
            # Copy EMA parameters to model
            self._copy_ema_to_model(pl_module)
            rank_zero_info("Applied EMA weights for validation")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore original weights after validation."""
        if self.ema_model is not None and self.validate_with_ema:
            # Restore original parameters
            self._restore_original_params(pl_module)
            rank_zero_info("Restored original weights after validation")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Apply EMA weights for testing."""
        if self.ema_model is not None and self.validate_with_ema:
            self._store_original_params(pl_module)
            self._copy_ema_to_model(pl_module)
            rank_zero_info("Applied EMA weights for testing")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore original weights after testing."""
        if self.ema_model is not None and self.validate_with_ema:
            self._restore_original_params(pl_module)
            rank_zero_info("Restored original weights after testing")

    def _store_original_params(self, pl_module: LightningModule) -> None:
        """Store current model parameters."""
        if not hasattr(self, '_original_state_dict'):
            self._original_state_dict = {}
        
        for name, param in pl_module.named_parameters():
            self._original_state_dict[name] = param.data.clone()
            
        if self.use_buffers:
            for name, buffer in pl_module.named_buffers():
                self._original_state_dict[name] = buffer.data.clone()

    def _copy_ema_to_model(self, pl_module: LightningModule) -> None:
        """Copy EMA parameters to the model."""
        ema_state_dict = self.ema_model.state_dict()
        
        for name, param in pl_module.named_parameters():
            if name in ema_state_dict:
                param.data.copy_(ema_state_dict[name])
                
        if self.use_buffers:
            for name, buffer in pl_module.named_buffers():
                if name in ema_state_dict:
                    buffer.data.copy_(ema_state_dict[name])

    def _restore_original_params(self, pl_module: LightningModule) -> None:
        """Restore original model parameters."""
        if hasattr(self, '_original_state_dict'):
            for name, param in pl_module.named_parameters():
                if name in self._original_state_dict:
                    param.data.copy_(self._original_state_dict[name])
                    
            if self.use_buffers:
                for name, buffer in pl_module.named_buffers():
                    if name in self._original_state_dict:
                        buffer.data.copy_(self._original_state_dict[name])

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        if self.ema_model is not None and self.save_ema_state:
            return {
                "ema_model_state": self.ema_model.state_dict(),
                "decay": self.decay,
                "use_buffers": self.use_buffers,
            }
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        if "ema_model_state" in state_dict and self.ema_model is not None:
            self.ema_model.load_state_dict(state_dict["ema_model_state"])
            rank_zero_info("Loaded EMA state from checkpoint")

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save EMA state in checkpoint."""
        if self.save_ema_state:
            checkpoint["ema_callback"] = self.state_dict()
        return checkpoint

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        """Load EMA state from checkpoint."""
        if "ema_callback" in checkpoint:
            self.load_state_dict(checkpoint["ema_callback"])

    @property
    def averaged_model(self) -> Optional[AveragedModel]:
        """Access to the underlying EMA model for inspection."""
        return self.ema_model