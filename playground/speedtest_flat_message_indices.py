import argparse
import time

import torch

from nets.equiformer_v2.hessian_pred_utils import (
    _get_flat_indexadd_message_indices_slow,
    _get_flat_indexadd_message_indices,
)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_edge_index(num_atoms: int, avg_degree: int, device: torch.device) -> torch.Tensor:
    # Approximate number of directed edges
    num_edges = max(1, int(num_atoms * avg_degree))
    # Sample random (i, j) with replacement. Exclude self-loops for realism.
    i = torch.randint(0, num_atoms, (num_edges,), device=device, dtype=torch.long)
    j = torch.randint(0, num_atoms, (num_edges,), device=device, dtype=torch.long)
    mask = i != j
    if not torch.all(mask):
        i = i[mask]
        j = j[mask]
    if i.numel() == 0:
        # fallback: create a trivial edge if everything filtered
        i = torch.tensor([0], device=device, dtype=torch.long)
        j = torch.tensor([min(1, num_atoms - 1)], device=device, dtype=torch.long)
    edge_index = torch.stack([i, j], dim=0)
    return edge_index


def time_once_slow(num_atoms: int, edge_index: torch.Tensor) -> float:
    device = edge_index.device
    synchronize_if_needed(device)
    start = time.time()
    _get_flat_indexadd_message_indices_slow(num_atoms, edge_index)
    synchronize_if_needed(device)
    end = time.time()
    return (end - start) * 1000.0, edge_index.shape[1]


def run_benchmark(
    sizes: list[int], avg_degree: int, device: torch.device, warmup: int = 3, repeats: int = 10
) -> None:
    print(f"Device: {device}")
    print("N\tE\tslow (ms)\tfast (ms)\tequal")
    for num_atoms in sizes:
        edge_index = make_edge_index(num_atoms, avg_degree, device)
        # correctness check
        ij_slow, ji_slow = _get_flat_indexadd_message_indices_slow(num_atoms, edge_index)
        ij_fast, ji_fast = _get_flat_indexadd_message_indices(num_atoms, edge_index)
        is_equal = torch.equal(ij_slow, ij_fast) and torch.equal(ji_slow, ji_fast)
        if not is_equal:
            raise AssertionError(
                f"Mismatch between slow and fast message index builders for N={num_atoms}"
            )
        # warmup
        for _ in range(warmup):
            time_once_slow(num_atoms, edge_index)
            synchronize_if_needed(device)
            _get_flat_indexadd_message_indices(num_atoms, edge_index)
        # measure
        times_ms_slow = []
        times_ms_fast = []
        E = edge_index.shape[1]
        for _ in range(repeats):
            ms_slow, _ = time_once_slow(num_atoms, edge_index)
            times_ms_slow.append(ms_slow)
            synchronize_if_needed(device)
            start = time.time()
            _get_flat_indexadd_message_indices(num_atoms, edge_index)
            synchronize_if_needed(device)
            times_ms_fast.append((time.time() - start) * 1000.0)
        avg_slow = sum(times_ms_slow) / len(times_ms_slow)
        avg_fast = sum(times_ms_fast) / len(times_ms_fast)
        print(f"{num_atoms}\t{E}\t{avg_slow:.3f}\t{avg_fast:.3f}\t{is_equal}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Speed test for _get_flat_indexadd_message_indices_slow (builds edge message indices)."
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
        "--avg_degree",
        type=int,
        default=8,
        help="Average out-degree per node used to sample edges.",
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
    """
    uv run /ssd/Code/gad-ff/playground/speedtest_flat_message_indices.py --device cuda --sizes 64 128 256 512 --avg_degree 12 --repeats 10
    """
    args = parse_args()
    device = torch.device(args.device)
    run_benchmark(args.sizes, args.avg_degree, device, warmup=args.warmup, repeats=args.repeats)


