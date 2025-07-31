import hydra
import re
from omegaconf import DictConfig
from gadff.logging_utils import name_from_config, find_latest_checkpoint


def setup_training(cfg: DictConfig):

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

    print(f"\n{cfg.ckpt_trainer_path}")

    return cfg.ckpt_trainer_path


@hydra.main(version_base=None, config_path="../configs", config_name="train_eigen")
def main(cfg: DictConfig) -> None:
    return setup_training(cfg)


if __name__ == "__main__":
    """Try:
    python scripts/train_eigen.py experiment=debug
    python scripts/train_eigen.py training.bz=2
    """
    main()
