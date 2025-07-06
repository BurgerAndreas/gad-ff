import numpy as np
import torch

num_atoms = 5
L = 9
C = 8

# Atom element symbols for 5 different atoms
atom_symbols = np.array(["C", "H", "N", "O", "S"], dtype=str)

# Create N,3 numpy array of strings for xyz coordinates of each atom
atom_xyz_labels = np.array(
    [[f"{i}_x", f"{i}_y", f"{i}_z"] for i in range(len(atom_symbols))]
)
print("atom_xyz_labels", atom_xyz_labels)

# Create Hessian matrix (N, 3, N, 3) of strings for pairwise interactions
coord_names = ['x', 'y', 'z']
hessian_strings = np.empty((num_atoms, 3, num_atoms, 3), dtype=object)

for i in range(num_atoms):
    for coord_i in range(3):
        for j in range(num_atoms):
            for coord_j in range(3):
                atom_i = i + 1  # atoms numbered 1 to 5
                atom_j = j + 1
                hessian_strings[i, coord_i, j, coord_j] = f"{atom_i}{coord_names[coord_i]}_{atom_j}{coord_names[coord_j]}"

print("Hessian shape:", hessian_strings.shape)
# i = 0 # atom1
# j = 1 # atom2
i = 2 # atom3
j = 1 # atom2
coord_i = 0 # x
coord_j = 2 # z
print(f"hessian atom{i+1}_x to atom{j+1}_z = {hessian_strings[i, coord_i, j, coord_j]}")  # atom1_x to atom2_z

# reshape to N*3, N*3
# '1x_1x' '1x_1y' '1x_1z' '1x_2x' '1x_2y', ...
# '1y_1x' '1y_1y' '1y_1z' '1y_2x' '1y_2y', ...
# '1z_1x' '1z_1y' '1z_1z' '1z_2x' '1z_2y', ...
# '2x_1x' '2x_1y' '2x_1z' '2x_2x' '2x_2y', ...
print("\n3N, 3N")
hessian_strings = hessian_strings.reshape(num_atoms*3, num_atoms*3)
print("hessian_strings 3N, 3N\n", hessian_strings[:4, :5])
index = (i * 3 + coord_i, j * 3 + coord_j)
print(f"hessian atom1_x to atom2_z = {hessian_strings[index]}")  # atom1_x to atom2_z

# reshape to a vector N*3*N*3
# '1x_1x' '1x_1y' '1x_1z' '1x_2x' '1x_2y' '1x_2z' '1x_3x' '1x_3y' '1x_3z'
# '1x_4x' '1x_4y' '1x_4z' '1x_5x' '1x_5y' '1x_5z' '1y_1x'
print("\nN*3*N*3")
hessian_strings = hessian_strings.reshape(-1)
print("hessian_strings\n", hessian_strings[:16])
# Calculate index for flattened array: i * (3 * N * 3) + coord_i * (N * 3) + j * 3 + coord_j
index = i * (3 * num_atoms * 3) + coord_i * (num_atoms * 3) + j * 3 + coord_j
hessian_element = hessian_strings[index]
print(f"hessian element = {hessian_element}")


# how I think index_add works
E = 10
D = 5
messsages = torch.randn(E)
hessian = torch.zeros(D)
indices = torch.randint(0, D, (E,)) # random indices from src -> dst
hessian.index_add_(0, indices, messsages)
hessian_loop = torch.zeros(D)
for i in range(E):
    hessian_loop[indices[i]] += messsages[i]
assert torch.allclose(hessian, hessian_loop)



# edge_index = torch.tensor(
#     [[0, 1, 1, 1, 2, 2, 3, 3], [1, 0, 2, 3, 1, 3, 1, 2]]
# )  # [2, E]

# E = edge_index.shape[1]

# l012_features = torch.randn(E, L, C)
# l012_features_rev = torch.randn(E, L, C)

# sym_message = (l012_features + l012_features_rev) / 2  # (E, 3, 3)

# hessian = torch.zeros((num_atoms, 3, num_atoms, 3))
# for ij in range(edge_index.shape[1]):
#     i, j = edge_index[0, ij], edge_index[1, ij]
#     hessian[i, :, j, :] = sym_message[ij]
#     hessian[j, :, i, :] = sym_message[ij].T

# # combine message with node embeddings (self-connection)
# # node embeddings -> (N, 3, 3)
# l012_node_features = torch.randn(num_atoms, 3, 3)
# # add node embeddings to diagonal of hessian
# for ii in range(num_atoms):
#     hessian[ii, :, ii, :] = l012_node_features[ii]

# hessian = hessian.view(num_atoms * 3, num_atoms * 3)
