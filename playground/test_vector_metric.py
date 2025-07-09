import torch


def cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def abs_cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.abs(torch.dot(a, b) / (torch.norm(a) * torch.norm(b)))


def batch_similarity(a, b, data, lossfn):
    """Returns a scalar of similarity averaged over batches.
    lossfn should return a scalar, otherwise it will be averaged over all entries returned.
    """
    B = data.batch.max() + 1
    ptr = data.ptr
    natoms = data.natoms
    sim = []
    for _b in range(B):
        _start = ptr[_b] * 3
        _end = (ptr[_b] + natoms[_b]) * 3
        a_b = a[_start:_end]
        b_b = b[_start:_end]
        sim_b = lossfn(a_b, b_b)
        sim.append(sim_b)
    return torch.stack(sim).mean()


B = 2

N1 = 3
# random vector N1*3
v1pred = torch.randn(N1, 3).reshape(-1)
v1true = torch.randn(N1, 3).reshape(-1)

N2 = 4
# random vector N2*3
v2pred = torch.randn(N2, 3).reshape(-1)
v2true = torch.randn(N2, 3).reshape(-1)

print()
sim1 = cosine_similarity(v1pred, v1true)
sim2 = cosine_similarity(v2pred, v2true)
print("Average similarity(sim1 + sim2)/2 (what we want)", (sim1 + sim2) / 2)

vpred = torch.cat([v1pred, v2pred], dim=0)
vtrue = torch.cat([v1true, v2true], dim=0)

print()

sim12 = cosine_similarity(vpred, vtrue)
print("naive sim12 (wrong)", sim12)

#######################################################################
print()


class FakeData:
    pass


natoms = torch.tensor([N1, N2])
ptr = torch.cat([torch.tensor([0]), torch.cumsum(natoms, dim=0)])
ptr = ptr[:-1]
batch = []
for _b in range(B):
    batch.append(torch.full((natoms[_b],), _b))
batch = torch.cat(batch, dim=0)

data = FakeData()
data.natoms = natoms
data.ptr = ptr
data.batch = batch

print(
    "batch_similarity (correct)",
    batch_similarity(vpred, vtrue, data, cosine_similarity),
)

#######################################################################
print()

a = v1pred.reshape(-1)
b = v1true.reshape(-1)

norma1 = torch.norm(a)
norma2 = torch.sqrt(torch.pow(a, 2).sum())

print("norma1", norma1)
print("norma2", norma2)

#######################################################################
print()
print("=== Gradient flow test ===")

# Test gradient flow through batch_similarity
vpred_grad = torch.cat([v1pred, v2pred], dim=0).clone().detach().requires_grad_(True)
vtrue_grad = torch.cat([v1true, v2true], dim=0).clone().detach()

print("vpred_grad.requires_grad:", vpred_grad.requires_grad)
print("vpred_grad.grad before:", vpred_grad.grad)

# Compute batch similarity
loss = batch_similarity(vpred_grad, vtrue_grad, data, cosine_similarity)
print("loss:", loss)
print("loss.requires_grad:", loss.requires_grad)

# Backward pass
loss.backward()

print("Gradient shape:", vpred_grad.grad.shape if vpred_grad.grad is not None else None)
print(
    "Gradient norm:",
    torch.norm(vpred_grad.grad) if vpred_grad.grad is not None else None,
)

if vpred_grad.grad is not None:
    print("✅ Gradient flow successful - can be used as loss function")
else:
    print("❌ Gradient flow failed - cannot be used as loss function")


#######################################################################
print()
print("=== Matrix multiplication associativity test ===")

# test if evecs_true.T @ hessian_pred @ evecs_true == evecs_true.T @ (hessian_pred @ evecs_true)

# Create test matrices
n_modes = 5
n_atoms = 10
evecs_true = torch.randn(n_atoms * 3, n_atoms * 3)
hessian_pred = torch.randn(n_atoms * 3, n_atoms * 3)

print(f"evecs_true shape: {evecs_true.shape}")
print(f"hessian_pred shape: {hessian_pred.shape}")

# Test associativity: A.T @ B @ A == A.T @ (B @ A)
result1 = evecs_true.T @ hessian_pred @ evecs_true
result2 = evecs_true.T @ (hessian_pred @ evecs_true)

print(f"result1 shape: {result1.shape}")
print(f"result2 shape: {result2.shape}")

# Check if they are equal within numerical tolerance
are_equal = torch.allclose(result1, result2, rtol=1e-5, atol=1e-8)
print(f"Are results equal within tolerance: {are_equal}")

if are_equal:
    print("✅ Matrix multiplication associativity verified")
else:
    print("❌ Matrix multiplication associativity failed")
    max_diff = torch.max(torch.abs(result1 - result2))
    print(f"Maximum difference: {max_diff}")

#######################################################################
print()
print("=== Matrix multiplication associativity test ===")

# test if evecs_true.T @ hessian_pred @ evecs_true == evecs_true.T @ (hessian_pred @ evecs_true)

# Create test matrices
n_modes = 5
n_atoms = 10
k = 10
evecs_true = torch.randn(n_atoms * 3, k)
hessian_pred = torch.randn(n_atoms * 3, n_atoms * 3)

print(f"evecs_true shape: {evecs_true.shape}")
print(f"hessian_pred shape: {hessian_pred.shape}")

# Test associativity: A.T @ B @ A == A.T @ (B @ A)
result1 = evecs_true.T @ hessian_pred @ evecs_true
result2 = evecs_true.T @ (hessian_pred @ evecs_true)

print(f"result1 shape: {result1.shape}")
print(f"result2 shape: {result2.shape}")

# Check if they are equal within numerical tolerance
are_equal = torch.allclose(result1, result2, rtol=1e-5, atol=1e-8)
print(f"Are results equal within tolerance: {are_equal}")

if are_equal:
    print("✅ Matrix multiplication associativity verified")
else:
    print("❌ Matrix multiplication associativity failed")
    max_diff = torch.max(torch.abs(result1 - result2))
    print(f"Maximum difference: {max_diff}")

#######################################################################
print()
