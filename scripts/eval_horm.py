import torch
import argparse
from gadff.horm.eval_horm import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HORM model on dataset")
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/eqv2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--hessian_method",
        type=str,
        default=None,
        help="Hessian computation method",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name (e.g., ts1x-val.lmdb, ts1x_hess_train_big.lmdb, RGD1.lmdb)",
    )
    parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all samples)",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    checkpoint_path = args.ckpt_path
    lmdb_path = args.dataset
    max_samples = args.max_samples
    config_path = args.config_path
    hessian_method = args.hessian_method

    evaluate(lmdb_path, checkpoint_path, config_path, hessian_method, max_samples)
