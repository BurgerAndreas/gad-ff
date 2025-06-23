import os
import omegaconf
import numpy as np
import torch
from uuid import uuid4

# allows to load checkpoint with the same name
IGNORE_OVERRIDES = [
]

# some stuff is not relevant for the checkpoint
# e.g. inference kwargs
IGNORE_OVERRIDES_CHECKPOINT = [
]

REPLACE = {
    "+": "",
    "experiment=": "",
    "training.lr_schedule_type=": "lr=",
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
                for key, value in REPLACE.items():
                    override = arg.replace(key, value)
                override_names += " " + override
    except Exception as error:
        print("\nname_from_config() failed:", error)
        print("args:", args)
        raise error
    # logger.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    _name = mname + override_names
    print(f"Name{' checkpoint' if is_checkpoint_name else ''}: {_name}")
    return _name

# def name_from_config(cfg):
#     if isinstance(cfg.training.trn_path, str):
#         dataset_name = cfg.training.trn_path.split("/")[-1].split(".")[0].replace("-eigen", "")
#     else:
#         dataset_name = ""
#         for path in cfg.training.trn_path:
#             dataset_name += "-" + path.split("/")[-1].split(".")[0].replace("-eigen", "")
#     run_name = f"{cfg.version}-{cfg.experiment_name}-{dataset_name}-" + str(uuid4()).split("-")[-1]
#     return run_name

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
