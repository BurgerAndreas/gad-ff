import argparse
import time
import torch

from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.data import Batch
import copy

from gadff.horm.training_module import PotentialModule
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path
from nets.prediction_utils import compute_extra_props
from nets.equiformer_v2.hessian_pred_utils import add_extra_props_for_hessian_optimized, add_extra_props_for_hessian
from ocpmodels.hessian_graph_transform import HessianGraphTransform


def main():
    parser = argparse.ArgumentParser(description="Speed test for Equiformer forward pass")
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/eqv2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb", # ts1x-val-hesspred.lmdb
        help="Dataset file name",
    )
    # No per-sample iters; we synthesize batches for timing tests below
    args = parser.parse_args()

    torch.manual_seed(42)

    # checkpoint_path = args.ckpt_path
    checkpoint_path = "/ssd/Code/ReactBench/ckpt/hesspred/alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956.ckpt"

    lmdb_path = args.dataset
    # Helper to build a synthetic batch by repeating the first dataset sample bz times
    def build_batch(bz: int):
        # base = dataset_yes[0]
        # data_list = [copy.deepcopy(base) for _ in range(bz)]
        # batched = Batch.from_data_list(data_list)
        batched = next(iter(TGDataLoader(
            dataset_yes, 
            batch_size=bz, 
            shuffle=False, 
            follow_batch=["diag_ij", "edge_index", "message_idx_ij"]
        )))
        batched = batched.to("cuda")
        batched = compute_extra_props(batched)
        return batched

    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]

    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to("cuda")
    model.eval()

    # Build datasets to measure HessianGraphTransform cost
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset_yes = LmdbDataset(fix_dataset_path(lmdb_path), transform=transform)
    dataset_no = LmdbDataset(fix_dataset_path(lmdb_path), transform=None)

    # Test 0: timing HessianGraphTransform using dataloaders with/without transform
    # HessianGraphTransform is extremely slow
    for _bz in [2, 4, 8, 16, 32]:
        dataloader_yes = TGDataLoader(dataset_yes, batch_size=_bz, shuffle=False, follow_batch=["diag_ij", "edge_index", "message_idx_ij"])
        dataloader_no = TGDataLoader(dataset_no, batch_size=_bz, shuffle=False, follow_batch=["diag_ij", "edge_index", "message_idx_ij"])
        K = 64
        # Measure no-transform loader fetch time
        times_no = []
        it_no = iter(dataloader_no)
        for i in range(K):
            t0 = time.perf_counter()
            _databatch = next(it_no)
            t1 = time.perf_counter()
            if i > 1: # first iteration is warmup
                times_no.append((t1 - t0) * 1000.0)
        # Measure with-transform loader fetch time
        times_yes = []
        it_yes = iter(dataloader_yes)
        for i in range(K):
            t0 = time.perf_counter()
            _databatch = next(it_yes)
            t1 = time.perf_counter()
            if i > 1: # first iteration is warmup
                times_yes.append((t1 - t0) * 1000.0)
        avg_no = sum(times_no) / len(times_no)
        avg_yes = sum(times_yes) / len(times_yes)
        print(f"HessianGraphTransform timing (ms/sample) (bz={_bz}):")
        print(f"  dataloader without transform: {avg_no:.2f}")
        print(f"  dataloader with transform:    {avg_yes:.2f}")

    # Warmup single forward pass
    with torch.no_grad():
        batch = build_batch(1)
        _ = model.forward(batch, otf_graph=False, hessian=True, add_props=False)
    torch.cuda.synchronize()

    # Test 1: functional equivalence of add_extra_props_for_hessian vs optimized
    with torch.no_grad():
        test_batch = build_batch(4)
        # run reference
        ref = copy.deepcopy(test_batch)
        ref = add_extra_props_for_hessian(ref, offset_indices=True)
        # run optimized
        opt = copy.deepcopy(test_batch)
        opt = add_extra_props_for_hessian_optimized(opt, offset_indices=True)
        def _equal(a, b):
            return hasattr(ref, a) and hasattr(opt, a) and torch.equal(getattr(ref, a), getattr(opt, a))
        fields = [
            "ptr_1d_hessian",
            "message_idx_ij",
            "message_idx_ji",
            "diag_ij",
            "diag_ji",
            "node_transpose_idx",
        ]
        ok = all(_equal(f, f) for f in fields)
        print(f"\nEquivalence test (B=4): {'PASS' if ok else 'FAIL'}")
        if not ok:
            for f in fields:
                if hasattr(ref, f) and hasattr(opt, f):
                    same = torch.equal(getattr(ref, f), getattr(opt, f))
                    print(f" - {f}: {'ok' if same else 'diff'}")

    # Test 2: standalone timing of add_extra_props... for varying B
    bz_list = [32, 64, 128, 256]
    repeats = 3
    print("\nStandalone add_extra_props timing (ms):")
    for bz in bz_list:
        try:
            t_ref = []
            t_opt = []
            with torch.no_grad():
                for _ in range(repeats):
                    batch = build_batch(bz)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = add_extra_props_for_hessian(batch, offset_indices=True)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    t_ref.append((t1 - t0) * 1000.0)
                for _ in range(repeats):
                    batch = build_batch(bz)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = add_extra_props_for_hessian_optimized(batch, offset_indices=True)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    t_opt.append((t1 - t0) * 1000.0)
            print(
                f"  B={bz}: ref {sum(t_ref)/len(t_ref):.2f} | opt {sum(t_opt)/len(t_opt):.2f}"
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"  B={bz}: OOM, skipping")
            else:
                raise

    # Test 3: full forward timing with each add_extra_props variant
    bz_list_fwd = [2, 4, 8, 16]
    repeats = 3
    print("\nFull forward timing (ms):")
    for bz in bz_list_fwd:
        try:
            t_ref = []
            t_opt = []
            with torch.no_grad():
                for _ in range(repeats):
                    batch = build_batch(bz)
                    batch_ref = copy.deepcopy(batch)
                    batch_ref = add_extra_props_for_hessian(batch_ref, offset_indices=True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model.forward(batch_ref, otf_graph=False, hessian=True, add_props=False)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    t_ref.append((t1 - t0) * 1000.0)
                for _ in range(repeats):
                    batch = build_batch(bz)
                    batch_opt = copy.deepcopy(batch)
                    batch_opt = add_extra_props_for_hessian_optimized(batch_opt, offset_indices=True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model.forward(batch_opt, otf_graph=False, hessian=True, add_props=False)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    t_opt.append((t1 - t0) * 1000.0)
            print(
                f"  B={bz}: ref {sum(t_ref)/len(t_ref):.2f} | opt {sum(t_opt)/len(t_opt):.2f}"
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"  B={bz}: OOM, skipping")
            else:
                raise


if __name__ == "__main__":
    main()


