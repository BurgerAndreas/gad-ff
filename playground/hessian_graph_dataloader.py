import torch
from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.data import Batch
from nets.equiformer_v2.hessian_pred_utils import _get_flat_indexadd_message_indices
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20


class HessianGraphDataLoader(TGDataLoader):
    """
    Custom DataLoader that precomputes graph and hessian message indices for efficient batching.
    """

    def __init__(self, dataset, model_config, batch_size=1, shuffle=False, **kwargs):
        """
        Args:
            dataset: torch_geometric dataset
            model_config: Configuration dictionary for the EquiformerV2 model
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            **kwargs: Additional arguments for DataLoader
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        # Create a model instance for graph generation
        self.model = EquiformerV2_OC20(**model_config)

    def _collate_fn(self, batch_list):
        """
        Custom collate function that precomputes graph and hessian indices.
        """
        # First, precompute graph and indices for each sample
        processed_batch = []
        for data in batch_list:
            # Generate graph using the model's method
            (
                edge_index,
                edge_dist,
                distance_vec,
                cell_offsets,
                cell_offset_distances,
                neighbors,
            ) = self.model.generate_graph(data)

            # Store graph information in data object
            data.edge_index = edge_index
            data.edge_dist = edge_dist
            data.edge_distance_vec = distance_vec
            data.cell_offsets = cell_offsets
            data.cell_offset_distances = cell_offset_distances
            data.neighbors = neighbors

            # Precompute hessian message indices
            N = data.pos.shape[0]  # Number of atoms
            indices_ij, indices_ji = _get_flat_indexadd_message_indices(N, edge_index)

            # Store indices in data object
            data.message_idx_ij = indices_ij
            data.message_idx_ji = indices_ji

            processed_batch.append(data)

        # Use the default collate function to create the batch
        return Batch.from_data_list(processed_batch)

    def __iter__(self):
        """
        Iterator that applies the custom collate function.
        """
        for batch in super().__iter__():
            yield self._collate_fn(batch)

    def __len__(self):
        """
        Return the number of batches.
        """
        return super().__len__()


# Alternative implementation using the collate_fn parameter
def create_hessian_graph_collate_fn(model_config):
    """
    Factory function to create a collate function for precomputing graph and hessian indices.

    Args:
        model_config: Configuration dictionary for the EquiformerV2 model

    Returns:
        collate_fn: Function that can be passed to DataLoader
    """
    # Create a model instance for graph generation
    model = EquiformerV2_OC20(**model_config)

    def collate_fn(batch_list):
        """
        Custom collate function that precomputes graph and hessian indices.
        """
        # First, precompute graph and indices for each sample
        processed_batch = []
        for data in batch_list:
            # Generate graph using the model's method
            (
                edge_index,
                edge_dist,
                distance_vec,
                cell_offsets,
                cell_offset_distances,
                neighbors,
            ) = model.generate_graph(data)

            # Store graph information in data object
            data.edge_index = edge_index
            data.edge_dist = edge_dist
            data.edge_distance_vec = distance_vec
            data.cell_offsets = cell_offsets
            data.cell_offset_distances = cell_offset_distances
            data.neighbors = neighbors

            # Precompute hessian message indices
            N = data.pos.shape[0]  # Number of atoms
            indices_ij, indices_ji = _get_flat_indexadd_message_indices(N, edge_index)

            # Store indices in data object
            data.message_idx_ij = indices_ij
            data.message_idx_ji = indices_ji

            processed_batch.append(data)

        # Use the default collate function to create the batch
        return Batch.from_data_list(processed_batch)

    return collate_fn
