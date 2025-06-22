"""Script to train new prediction heads for Hessian eigenvalues and eigenvectors.

Starts from the checkpoint of the EquiformerV2 model finetuned on the HORM dataset.
Keeps the existing weights frozen.
Adds one extra head each to predict the smallest two eigenvalues and eigenvectors of the Hessian.
"""
from uuid import uuid4
from copy import deepcopy
import yaml
import torch
import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from gadff.eigen_training_module import EigenPotentialModule
from gadff.path_config import DATASET_DIR_HORM_EIGEN, DATASET_FILES_HORM, CHECKPOINT_PATH_EQUIFORMER_HORM

def setup_training(cfg: DictConfig):
    run_name = f"{cfg.version}-" + str(uuid4()).split("-")[-1]

    # from the paper:
    # Model Layers HiddenDim Heads LearningRate BatchSize
    # EquiformerV2 4 128 4 3e-4 128
    # maximum spherical harmonic degree of lmax = 4
    # with open("configs/equiformer_v2.yml", "r") as f:
    #     model_config = yaml.safe_load(f)

    # add extra heads to predict Hessian eigenvalues and eigenvectors
    # Add eigenvalue/eigenvector prediction flags to model config
    # model_config.update({
    #     'do_eigvec_1': cfg.do_eigvec_1,
    #     'do_eigvec_2': cfg.do_eigvec_2,
    #     'do_eigval_1': cfg.do_eigval_1,
    #     'do_eigval_2': cfg.do_eigval_2,
    # })
    
    model_config = cfg.model
    print("Model config:\n", yaml.dump(model_config))

    optimizer_config = dict(cfg.optimizer)

    training_config = dict(cfg.training)

    pm = EigenPotentialModule(model_config, optimizer_config, training_config)

    logger = wandb_logger = WandbLogger(
        project=cfg.project,
        log_model=False,
        name=run_name,
    )

    ckpt_output_path = f"checkpoint/{cfg.project}/{wandb_logger.experiment.name}"

    checkpoint_callback = ModelCheckpoint(
        monitor="val-totloss",
        dirpath=ckpt_output_path,
        filename="ff-{epoch:03d}-{val-totloss:.4f}-{val-MAE_E:.4f}-{val-MAE_F:.4f}",
        every_n_epochs=10,
        save_top_k=-1,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val-totloss",
        patience=1000,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        TQDMProgressBar(),
        lr_monitor,
    ]

    trainer = pl.Trainer(
        devices=cfg.pltrainer.devices,
        num_nodes=cfg.pltrainer.num_nodes,
        accelerator=cfg.pltrainer.accelerator,
        strategy=cfg.pltrainer.strategy,
        max_epochs=cfg.pltrainer.max_epochs,
        callbacks=callbacks,
        # path for logs and weights when no logger/ckpt_callback passed
        default_root_dir=ckpt_output_path,
        logger=logger,
        gradient_clip_val=cfg.pltrainer.gradient_clip_val,
        accumulate_grad_batches=cfg.pltrainer.accumulate_grad_batches,
        limit_train_batches=cfg.pltrainer.limit_train_batches,
        limit_val_batches=cfg.pltrainer.limit_val_batches,
    )
    return trainer, pm

@hydra.main(version_base=None, config_path="../configs", config_name="train_eigen")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    trainer, pm = setup_training(cfg)
    trainer.fit(pm, ckpt_path=CHECKPOINT_PATH_EQUIFORMER_HORM)

if __name__ == "__main__":
    main()
