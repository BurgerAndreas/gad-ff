import numpy as np
import torch

num_atoms = 4
L = 9
C = 8

edge_index = torch.tensor(
    [[0, 1, 1, 1, 2, 2, 3, 3], [1, 0, 2, 3, 1, 3, 1, 2]]
)  # [2, E]

E = edge_index.shape[1]

l012_features = torch.randn(E, L, C)
l012_features_rev = torch.randn(E, L, C)

sym_message = (l012_features + l012_features_rev) / 2  # (E, 3, 3)

hessian = torch.zeros((num_atoms, 3, num_atoms, 3))
for ij in range(edge_index.shape[1]):
    i, j = edge_index[0, ij], edge_index[1, ij]
    hessian[i, :, j, :] = sym_message[ij]
    hessian[j, :, i, :] = sym_message[ij].T

# combine message with node embeddings (self-connection)
# node embeddings -> (N, 3, 3)
l012_node_features = torch.randn(num_atoms, 3, 3)
# add node embeddings to diagonal of hessian
for ii in range(num_atoms):
    hessian[ii, :, ii, :] = l012_node_features[ii]

hessian = hessian.view(num_atoms * 3, num_atoms * 3)
