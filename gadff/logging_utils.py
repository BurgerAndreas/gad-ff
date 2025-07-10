import os
import omegaconf
import numpy as np
import torch
from uuid import uuid4

# allows to load checkpoint with the same name
IGNORE_OVERRIDES = []

# some stuff is not relevant for the checkpoint
# e.g. inference kwargs
IGNORE_OVERRIDES_CHECKPOINT = []

REPLACE = {
    "+": "",
    "experiment=": "",
    "experiment": "",
    "training.": "",
    "training.lr_schedule_type=": "lr=",
    "training.eigen_loss": "el",
    "loss_type_vec=": "lossvec=",
    "loss_type=": "loss=",
}


def name_from_config(args: omegaconf.DictConfig, is_checkpoint_name=False) -> str:
    """Generate a name for the model based on the config.
    Name is intended to be used as a file name for saving checkpoints and outputs.
    """
    try:
        # model name format:
        mname = ""
        # override format: 'pretrain_dataset=bridge,steps=10,use_wandb=False'
        override_names = ""
        # print(f'Overrides: {args.override_dirname}')
        if args.override_dirname:
            for arg in args.override_dirname.split(","):
                # make sure we ignore some overrides
                if np.any([ignore in arg for ignore in IGNORE_OVERRIDES]):
                    continue
                if is_checkpoint_name:
                    if np.any(
                        [ignore in arg for ignore in IGNORE_OVERRIDES_CHECKPOINT]
                    ):
                        continue
                override_names += " " + arg
    except Exception as error:
        print("\nname_from_config() failed:", error)
        print("args:", args)
        raise error
    for key, value in REPLACE.items():
        override_names = override_names.replace(key, value)
    # logger.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    _name = mname + override_names
    print(f"Name{' checkpoint' if is_checkpoint_name else ''}: {_name}")
    return _name


def set_gpu_name(args):
    """Set wandb.run.name."""
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name = (
            gpu_name.replace("NVIDIA", "").replace("GeForce", "").replace(" ", "")
        )
        args.gpu_name = gpu_name
    except:
        pass
    return args
