import argparse
import time

import torch

from nets.equiformer_v2.hessian_pred_utils import (
    _get_node_diagonal_1d_indexadd_indices_slow,
    _get_node_diagonal_1d_indexadd_indices,
)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_once(num_atoms: int, device: torch.device) -> float:
    synchronize_if_needed(device)
    start = time.time()
    _get_node_diagonal_1d_indexadd_indices_slow(num_atoms, device)
    synchronize_if_needed(device)
    end = time.time()
    return (end - start) * 1000.0


def run_benchmark(
    sizes: list[int], device: torch.device, warmup: int = 3, repeats: int = 10
) -> None:
    print(f"Device: {device}")
    print("N\tslow (ms)\tfast (ms)\tequal")
    for num_atoms in sizes:
        # correctness check once per size
        d1, d2, t = _get_node_diagonal_1d_indexadd_indices_slow(num_atoms, device)
        f1, f2, ft = _get_node_diagonal_1d_indexadd_indices(num_atoms, device)
        is_equal = (
            torch.equal(d1, f1) and torch.equal(d2, f2) and torch.equal(t, ft)
        )
        if not is_equal:
            raise AssertionError(
                f"Mismatch between slow and fast index builders for N={num_atoms}"
            )
        # warmup
        for _ in range(warmup):
            time_once(num_atoms, device)
            synchronize_if_needed(device)
            _get_node_diagonal_1d_indexadd_indices(num_atoms, device)
        # measure
        slow_ms = [time_once(num_atoms, device) for _ in range(repeats)]
        synchronize_if_needed(device)
        start = time.time()
        for _ in range(repeats):
            _get_node_diagonal_1d_indexadd_indices(num_atoms, device)
        synchronize_if_needed(device)
        fast_ms = (time.time() - start) * 1000.0 / repeats
        print(f"{num_atoms}\t{sum(slow_ms)/len(slow_ms):.3f}\t{fast_ms:.3f}\t{is_equal}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Speed test for _get_node_diagonal_1d_indexadd_indices_slow (builds indices for"
            " node-diagonal index_add)."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cpu", "cuda"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="List of atom counts N to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs per size.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed runs per size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    run_benchmark(args.sizes, device, warmup=args.warmup, repeats=args.repeats)


