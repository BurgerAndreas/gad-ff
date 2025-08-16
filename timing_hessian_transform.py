#!/usr/bin/env python3
"""
Script to time various components of HessianGraphTransform to identify bottlenecks.
"""

import time
import torch
from torch_geometric.data import Data as TGData, Batch as TGBatch

# Add the project root to path so we can import modules
import sys
sys.path.append('/ssd/Code/gad-ff')

from ocpmodels.hessian_graph_transform import HessianGraphTransform, generate_fullyconnected_graph_nopbc
from nets.equiformer_v2.hessian_pred_utils import (
    _get_flat_indexadd_message_indices,
    _get_node_diagonal_1d_indexadd_indices,
)


def create_synthetic_batch(batch_size, atoms_per_mol=15, device='cpu'):
    """Create a synthetic batch for timing tests."""
    data_list = []
    for i in range(batch_size):
        # Random positions in a 10x10x10 box
        positions = torch.randn(atoms_per_mol, 3, device=device) * 5.0
        # Random atomic numbers (C, N, O)
        atomic_nums = torch.randint(6, 9, (atoms_per_mol,), device=device)
        
        data = TGData(
            pos=positions,
            z=atomic_nums,
            charges=atomic_nums,
            natoms=torch.tensor([atoms_per_mol], dtype=torch.int64, device=device),
            cell=torch.eye(3, device=device) * 10.0,
            pbc=torch.tensor([False, False, False], device=device),
        )
        data_list.append(data)
    
    # Create batch
    batch = TGBatch.from_data_list(data_list)
    return batch


def time_function(func, *args, **kwargs):
    """Time a function with synchronization."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    return result, end_time - start_time


def benchmark_graph_generation(batch, cutoff=6.0):
    """Time the graph generation part."""
    def generate_graphs():
        return generate_fullyconnected_graph_nopbc(batch, cutoff=cutoff)
    
    result, elapsed = time_function(generate_graphs)
    return result, elapsed


def benchmark_message_indices(batch):
    """Time the message index computation."""
    # First generate a graph to get edge_index
    edge_index, _, _, _, _, _ = generate_fullyconnected_graph_nopbc(batch, cutoff=6.0)
    N = batch.natoms.sum().item()
    
    def compute_message_indices():
        return _get_flat_indexadd_message_indices(N, edge_index)
    
    result, elapsed = time_function(compute_message_indices)
    return result, elapsed


def benchmark_diagonal_indices(batch):
    """Time the diagonal index computation."""
    N = batch.natoms.sum().item()
    device = batch.pos.device
    
    def compute_diagonal_indices():
        return _get_node_diagonal_1d_indexadd_indices(N, device)
    
    result, elapsed = time_function(compute_diagonal_indices)
    return result, elapsed


def benchmark_full_transform(batch, cutoff=6.0):
    """Time the full HessianGraphTransform."""
    transform = HessianGraphTransform(cutoff=cutoff, max_neighbors=None, use_pbc=False)
    
    def apply_transform():
        # Apply to each data in the batch separately (as done in training)
        data_list = batch.to_data_list()
        transformed_list = []
        for data in data_list:
            transformed_data = transform(data)
            transformed_list.append(transformed_data)
        return TGBatch.from_data_list(transformed_list)
    
    result, elapsed = time_function(apply_transform)
    return result, elapsed


def run_scaling_test(device):
    """Test how timing scales with batch size."""
    print()
    print("=" * 70)
    print("HESSIAN GRAPH TRANSFORM TIMING ANALYSIS")
    print(f"Device: {device}")
    print("=" * 70)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    atoms_per_mol = 15
    cutoff = 6.0
    
    results = {
        'batch_size': [],
        'graph_gen_time': [],
        'message_idx_time': [],
        'diagonal_idx_time': [],
        'full_transform_time': [],
        'total_atoms': [],
        'total_edges': []
    }
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size: {batch_size} ---")
        
        # Create synthetic batch
        batch = create_synthetic_batch(batch_size, atoms_per_mol, device)
        total_atoms = batch.natoms.sum().item()
        
        # Time graph generation
        graph_result, graph_time = benchmark_graph_generation(batch, cutoff)
        edge_index, _, _, _, _, _ = graph_result
        total_edges = edge_index.shape[1]
        
        # Time message index computation
        _, message_time = benchmark_message_indices(batch)
        
        # Time diagonal index computation
        _, diagonal_time = benchmark_diagonal_indices(batch)
        
        # Time full transform
        _, full_time = benchmark_full_transform(batch, cutoff)
        
        # Store results
        results['batch_size'].append(batch_size)
        results['graph_gen_time'].append(graph_time)
        results['message_idx_time'].append(message_time)
        results['diagonal_idx_time'].append(diagonal_time)
        results['full_transform_time'].append(full_time)
        results['total_atoms'].append(total_atoms)
        results['total_edges'].append(total_edges)
        
        # Print results
        print(f"  Total atoms: {total_atoms}")
        print(f"  Total edges: {total_edges}")
        print(f"  Graph generation: {graph_time:.4f}s")
        print(f"  Message indices:  {message_time:.4f}s")
        print(f"  Diagonal indices: {diagonal_time:.4f}s")
        print(f"  Full transform:   {full_time:.4f}s")
        print(f"  Per-sample transform: {full_time/batch_size:.4f}s")
    
    # Print scaling analysis
    print("\n" + "=" * 70)
    print(f"SCALING ANALYSIS on {device}")
    print("=" * 70)
    
    print(f"{'Batch Size':<10} {'Graph Gen':<10} {'Msg Idx':<10} {'Diag Idx':<10} {'Full Xform':<12} {'Per-Sample':<12}")
    print("-" * 70)
    
    # Print summary table
    for i, batch_size in enumerate(batch_sizes):
        per_sample_time = results['full_transform_time'][i] / batch_size
        print(f"{batch_size:<10} {results['graph_gen_time'][i]:<10.4f} {results['message_idx_time'][i]:<10.4f} "
              f"{results['diagonal_idx_time'][i]:<10.4f} {results['full_transform_time'][i]:<12.4f} {per_sample_time:<12.4f}")
    
    # Discard batch size 1, then fit a linear model to the remaining data
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Filter out batch size 1 for linear regression analysis
    filtered_indices = [i for i, bs in enumerate(batch_sizes) if bs > 1]
    
    if len(filtered_indices) > 1:
        print(f"\n{'='*70}")
        print("LINEAR SCALING ANALYSIS on {device} (excluding batch size 1)")
        print(f"{'='*70}")
        
        # Extract data for regression (excluding batch size 1)
        x_data = np.array([results['total_atoms'][i] for i in filtered_indices]).reshape(-1, 1)
        
        # Analyze scaling for each timing component
        components = {
            'Graph Generation': [results['graph_gen_time'][i] for i in filtered_indices],
            'Message Indices': [results['message_idx_time'][i] for i in filtered_indices],
            'Diagonal Indices': [results['diagonal_idx_time'][i] for i in filtered_indices],
            'Full Transform': [results['full_transform_time'][i] for i in filtered_indices]
        }
        
        print(f"{'Component':<18} {'Slope (s/atom)':<15} {'Intercept (s)':<15} {'R²':<10}")
        print("-" * 70)
        
        for comp_name, y_data in components.items():
            y_array = np.array(y_data)
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(x_data, y_array)
            
            # Calculate R²
            y_pred = reg.predict(x_data)
            ss_res = np.sum((y_array - y_pred) ** 2)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"{comp_name:<18} {reg.coef_[0]:<15.6f} {reg.intercept_:<15.6f} {r_squared:<10.3f}")
        
        # Analysis of per-atom scaling
        print(f"\n{'='*50}")
        print(f"PER-ATOM SCALING ANALYSIS on {device}")
        print(f"{'='*50}")
        
        per_atom_times = [results['full_transform_time'][i] / results['total_atoms'][i] for i in filtered_indices]
        avg_per_atom = np.mean(per_atom_times)
        std_per_atom = np.std(per_atom_times)
        
        print(f"Average time per atom: {avg_per_atom:.6f} ± {std_per_atom:.6f} seconds")
        
        batch_sizes_filtered = [batch_sizes[i] for i in filtered_indices]
        print(f"Per-atom times by batch size:")
        for bs, per_atom in zip(batch_sizes_filtered, per_atom_times):
            print(f"  Batch size {bs}: {per_atom:.6f} s/atom")
    
    return results



if __name__ == "__main__":
    # Run the benchmarks
    
    run_scaling_test(device="cpu")
    
    run_scaling_test(device="cuda")

