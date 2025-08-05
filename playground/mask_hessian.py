import torch

# also include the diagonal
hessian = torch.ones((3, 3)).triu(diagonal=0)

# hessian += 2 * torch.ones((3, 3)).tril(diagonal=0)

print(hessian)
