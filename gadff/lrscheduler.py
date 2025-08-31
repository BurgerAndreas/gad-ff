import math
import types
import warnings
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    cast,
    Literal,
    Optional,
    SupportsFloat,
    TypedDict,
    Union,
)
from weakref import ref

from torch import inf, Tensor
import os
import torch
import matplotlib.pyplot as plt

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import _warn_get_lr_called_within_step

class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.
    The only difference from the PyTorch StepLR is that it supports a minimum learning rate.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.last_epoch specifies the index of the previous epoch. 
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of the previous epoch. Default: -1.
        min_lr (float): Minimum learning rate. Default: None.
        
    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        min_lr: Optional[float] = None,
        last_epoch: int = -1,
    ):  # noqa: D107
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr 
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            lrs = [group["lr"] for group in self.optimizer.param_groups]
        else:
            lrs = [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        if self.min_lr is not None:
            lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs

    def _get_closed_form_lr(self):
        lrs = [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]
        if self.min_lr is not None:
            lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs

if __name__ == "__main__":
    # plot learning rate schedule
    # Create a tiny model and two independent optimizers
    model1 = torch.nn.Linear(1, 1)
    model2 = torch.nn.Linear(1, 1)
    model3 = torch.nn.Linear(1, 1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.05)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.05)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.05)
    # When resuming (last_epoch >= 0), PyTorch expects 'initial_lr' in param groups
    for opt in [optimizer1, optimizer2, optimizer3]:
        for group in opt.param_groups:
            group.setdefault("initial_lr", group["lr"])  # base LR before any scheduling

    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.9)
    scheduler2 = StepLR(optimizer2, step_size=2, gamma=0.9, last_epoch=10)
    scheduler3 = StepLR(optimizer3, step_size=2, gamma=0.9, lrmin=0.02)

    num_epochs = 40
    lrs1 = []
    lrs2 = []
    lrs3 = []
    epochs = list(range(num_epochs))
    for _ in epochs:
        lrs1.append(optimizer1.param_groups[0]["lr"]) 
        lrs2.append(optimizer2.param_groups[0]["lr"]) 
        lrs3.append(optimizer3.param_groups[0]["lr"])
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

    plt.plot(epochs, lrs1, label="scheduler1 (last_epoch=-1)")
    plt.plot(epochs, lrs2, label="scheduler2 (last_epoch=10)")
    plt.plot(epochs, lrs3, label=f"scheduler3 (last_epoch=-1, lrmin=0.02)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("StepLR schedule comparison")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    fname = "playground/plots/step_lr_schedule_comparison.png"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname)
    print(f"Saved plot to {fname}")