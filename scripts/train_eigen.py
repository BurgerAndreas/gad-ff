"""Script to train new prediction heads for Hessian eigenvalues and eigenvectors.

Starts from the checkpoint of the EquiformerV2 model finetuned on the HORM dataset.
Keeps the existing weights frozen.
Adds one extra head each to predict the smallest two eigenvalues and eigenvectors of the Hessian.
"""

import os
import torch
import hydra
import re
from omegaconf import DictConfig
import wandb
from datetime import datetime
from pathlib import Path

from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from gadff.training_module_eigen import EigenPotentialModule, MyPLTrainer
from gadff.training_module_hessian import HessianPotentialModule, MyPLTrainer
from gadff.path_config import CHECKPOINT_PATH_EQUIFORMER_HORM
from gadff.logging_utils import name_from_config, find_latest_checkpoint


def setup_training(cfg: DictConfig):
    run_name = name_from_config(cfg)

    # from the HORM paper:
    # Model Layers HiddenDim Heads LearningRate BatchSize
    # EquiformerV2 4 128 4 3e-4 128
    # maximum spherical harmonic degree of lmax = 4
    # with open("configs/equiformer_v2.yaml", "r") as f:
    #     model_config = yaml.safe_load(f)
    model_config = cfg.model
    # print("Model config:\n", yaml.dump(model_config))

    optimizer_config = dict(cfg.optimizer)
    training_config = dict(cfg.training)

    # Add SLURM job ID to config if it exists in environment
    if "SLURM_JOB_ID" in os.environ:
        cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]
        training_config["trn_slurm_job_id"] = cfg.slurm_job_id
    print(f"SLURM job ID: {cfg.slurm_job_id}")

    ###########################################
    # Model checkpoint loading
    ###########################################
    # ckpt_model_path
    # only loads the model weights, not the trainer state
    # like optimizer, learning rate scheduler, epoch/step, RNG state, etc.

    # pm = EigenPotentialModule(model_config, optimizer_config, training_config)
    # pm = hydra.utils.instantiate(cfg.potential_module_class, model_config, optimizer_config, training_config)
    pm = eval(cfg.potential_module_class)(
        model_config, optimizer_config, training_config
    )
    if cfg.ckpt_model_path == "horm":
        ckpt = torch.load(
            CHECKPOINT_PATH_EQUIFORMER_HORM, map_location="cuda", weights_only=True
        )
        print(f"Checkpoint keys: {ckpt.keys()}")
        print(f"Checkpoint state_dict keys: {len(ckpt['state_dict'].keys())}")
        # keys all start with `potential.`
        state_dict = {
            k.replace("potential.", ""): v for k, v in ckpt["state_dict"].items()
        }
        pm.potential.load_state_dict(state_dict, strict=False)
    elif os.path.exists(cfg.ckpt_model_path):
        # pm = hydra.utils.instantiate(cfg.potential_module_class).load_from_checkpoint(
        pm = eval(cfg.potential_module_class).load_from_checkpoint(
            cfg.ckpt_model_path, strict=False
        )
    else:
        print(f"Not loading model checkpoint from {cfg.ckpt_model_path}")
    print(f"{cfg.potential_module_class} initialized")

    ###########################################
    # Trainer checkpoint loading
    ###########################################
    # get checkpoint name
    run_name_ckpt = name_from_config(cfg, is_checkpoint_name=True)
    checkpoint_name = re.sub(r"[^a-zA-Z0-9]", "", run_name_ckpt)
    if len(checkpoint_name) <= 1:
        checkpoint_name = "base"
    print(f"Checkpoint name: {checkpoint_name}")

    # Auto-resume logic: find existing trainer checkpoint with same base name
    if cfg.get("ckpt_resume_auto", False):
        if cfg.ckpt_trainer_path is not None:
            print(
                f"Auto-resume is overwriting ckpt_trainer_path: {cfg.ckpt_trainer_path}"
            )
        print("Auto-resume enabled, searching for existing checkpoints...")
        latest_ckpt = find_latest_checkpoint(checkpoint_name, cfg.project)
        if latest_ckpt:
            cfg.ckpt_trainer_path = latest_ckpt
            print(f"Auto-resume: Will resume from {latest_ckpt}")
        else:
            print("Auto-resume: No existing checkpoints found, starting fresh")

    if cfg.ckpt_trainer_path is not None and cfg.ckpt_model_path is not None:
        # If both ckpt_model_path and ckpt_trainer_path are specified,
        # the ckpt_model_path loading becomes redundant
        # since those weights get immediately overwritten by the trainer checkpoint.
        print("Warning: ckpt_trainer_path will override ckpt_model_path")

    ###########################################
    # Trainer checkpoint saving
    ###########################################
    # add slurm job id and timestamp to checkpoint name
    checkpoint_name = f"{checkpoint_name}-{cfg.slurm_job_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Checkpoint name: {checkpoint_name}")

    ckpt_output_path = f"checkpoint/{cfg.project}/{checkpoint_name}"
    print(f"Checkpoint output path: {ckpt_output_path}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val-totloss",
        dirpath=ckpt_output_path,
        filename="ff-{epoch:03d}-{val-totloss:.4f}",
        every_n_epochs=10,
        save_top_k=2,
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

    wandb_kwargs = {}
    if not cfg.use_wandb:
        wandb_kwargs["mode"] = "disabled"
    wandb_logger = WandbLogger(
        project=cfg.project,
        log_model=False,
        name=run_name,
        **wandb_kwargs,
    )

    print("Initializing trainer")
    # trainer = pl.Trainer(
    trainer = MyPLTrainer(
        devices=cfg.pltrainer.devices,
        num_nodes=cfg.pltrainer.num_nodes,
        accelerator=cfg.pltrainer.accelerator,
        strategy=cfg.pltrainer.strategy,
        max_epochs=cfg.pltrainer.max_epochs,
        callbacks=callbacks,
        # path for logs and weights when no logger/ckpt_callback passed
        default_root_dir=ckpt_output_path,
        logger=wandb_logger,
        gradient_clip_val=cfg.pltrainer.gradient_clip_val,
        accumulate_grad_batches=cfg.pltrainer.accumulate_grad_batches,
        limit_train_batches=cfg.pltrainer.limit_train_batches,
        limit_val_batches=cfg.pltrainer.limit_val_batches,
        log_every_n_steps=cfg.pltrainer.log_every_n_steps,
        # check_val_every_n_epoch=cfg.pltrainer.get('check_val_every_n_epoch', 1),
        # val_check_interval=cfg.pltrainer.get('val_check_interval', None),
    )
    print("Trainer initialized")
    return trainer, pm


@hydra.main(version_base=None, config_path="../configs", config_name="train_eigen")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    trainer, pm = setup_training(cfg)
    print("Fitting model")
    trainer.fit(pm, ckpt_path=cfg.ckpt_trainer_path)


if __name__ == "__main__":
    """Try:
    python scripts/train_eigen.py experiment=debug
    python scripts/train_eigen.py training.bz=2
    """
    main()
