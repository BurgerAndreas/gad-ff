import torch
import torch.nn.functional as F
from functools import partial


def get_vector_loss_fn(loss_name: str, **kwargs):
    if loss_name == "cosine_squared":
        return partial(cosine_squared_loss, **kwargs)
    elif loss_name == "angle":
        return partial(L_ang_loss, **kwargs)
    elif loss_name == "cosine":
        return partial(cosine_loss, **kwargs)
    elif loss_name == "min_l2":
        return min_l2_loss
    elif loss_name == "min_l1":
        return min_l1_loss
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


def get_scalar_loss_fn(loss_name: str, **kwargs):
    if loss_name == "log_mse":
        return log_mse_loss
    elif loss_name == "huber":
        return HuberLoss(**kwargs)
    elif loss_name.lower() in ["mae", "l1"]:
        return F.l1_loss
    elif loss_name.lower() in ["mse", "l2"]:
        return F.mse_loss
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


##############################################################################
# vector losses


def cosine_similarity(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Sign-invariant cosine similarity loss: |cos(pred, target)|"""
    B = pred.shape[0]
    pred = pred.view(B, -1)
    target = target.view(B, -1)
    pred_norm = pred / (torch.norm(pred, dim=-1, keepdim=True) + eps)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + eps)
    cosine_sim = torch.sum(pred_norm * target_norm, dim=-1)
    return torch.mean(torch.abs(cosine_sim))


def cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Sign-invariant cosine similarity loss: 1 - |cos(pred, target)|"""
    B = pred.shape[0]
    pred = pred.view(B, -1)
    target = target.view(B, -1)
    pred_norm = pred / (torch.norm(pred, dim=-1, keepdim=True) + eps)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + eps)
    cosine_sim = torch.sum(pred_norm * target_norm, dim=-1)
    return torch.mean(1.0 - torch.abs(cosine_sim))


def min_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minimum L2 loss between pred vs target and pred vs -target"""
    B = pred.shape[0]
    pred = pred.view(B, -1)
    target = target.view(B, -1)
    loss_pos = torch.mean((pred - target) ** 2, dim=-1)
    loss_neg = torch.mean((pred + target) ** 2, dim=-1)
    return torch.mean(torch.min(loss_pos, loss_neg))


def min_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minimum L1 loss between pred vs target and pred vs -target"""
    B = pred.shape[0]
    pred = pred.view(B, -1)
    target = target.view(B, -1)
    loss_pos = torch.mean(torch.abs(pred - target), dim=-1)
    loss_neg = torch.mean(torch.abs(pred + target), dim=-1)
    return torch.mean(torch.min(loss_pos, loss_neg))


def cosine_squared_loss(
    v_pred: torch.Tensor,
    v_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cosine-squared loss, sign-invariant:
      L = 1 - (v_pred · v_true)^2

    Args:
      v_pred: Tensor of shape (B, N, 3) or (B, N*3)
      v_true: Tensor of shape (B, N, 3) or (B, N*3)
      eps:     small constant to avoid NaNs

    Returns:
      scalar tensor: average loss over batch and N
    """
    B = v_pred.shape[0]
    v_pred = v_pred.view(B, -1)
    v_true = v_true.view(B, -1)
    # normalize to unit vectors
    v_pred_norm = F.normalize(v_pred, dim=-1, eps=eps)
    v_true_norm = F.normalize(v_true, dim=-1, eps=eps)

    # dot product along last dim → shape (B)
    dots = torch.sum(v_pred_norm * v_true_norm, dim=-1)
    assert dots.shape == (
        B,
    ), f"dots.shape: {dots.shape}, v_pred.shape: {v_pred.shape}, v_true.shape: {v_true.shape}"

    # cosine-squared loss
    loss = 1.0 - dots.pow(2)

    return loss.mean()


def L_ang_loss(
    v_pred: torch.Tensor,
    v_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Squared angle loss, sign-invariant:
      L = arccos(|v_pred · v_true|)^2

    Args:
      v_pred: Tensor of shape (B, N, 3)
      v_true: Tensor of shape (B, N, 3)
      eps:     small constant to clamp dot into [-1, 1]

    Returns:
      scalar tensor: average loss over batch and N
    """
    B = v_pred.shape[0]
    v_pred = v_pred.view(B, -1)
    v_true = v_true.view(B, -1)
    # normalize
    v_pred_norm = F.normalize(v_pred, dim=-1, eps=eps)
    v_true_norm = F.normalize(v_true, dim=-1, eps=eps)

    # dot product
    dots = torch.sum(v_pred_norm * v_true_norm, dim=-1).abs()

    # clamp for numeric stability
    dots = dots.clamp(-1.0 + eps, 1.0 - eps)

    # squared arccosine
    ang = torch.acos(dots)
    loss = ang.pow(2)

    return loss.mean()


##############################################################################
# scalar losses
def log_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the log-mean-squared-error:
        mean( (log(|pred| + ε) - log(|target| + ε))^2 )

    Supports pred/target of shape (B,), (B,1) or scalar ().
    """
    # squeeze any singleton trailing dimension
    if pred.dim() > 1 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.dim() > 1 and target.size(-1) == 1:
        target = target.squeeze(-1)

    # now pred and target should be same shape: either (B,) or ()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}")

    lp = torch.log(torch.abs(pred) + epsilon)
    lt = torch.log(torch.abs(target) + epsilon)
    return torch.mean((lp - lt) ** 2)


class HuberLoss(torch.nn.Module):
    """
    Huber Loss implementation for PyTorch.

    Combines MSE for small errors and MAE for large errors.
    Loss = 0.5 * (y_true - y_pred)^2                     if |y_true - y_pred| <= delta
    Loss = delta * |y_true - y_pred| - 0.5 * delta^2    if |y_true - y_pred| > delta

    Args:
        delta (float): Threshold where loss transitions from quadratic to linear
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, delta=1.0, reduction="mean"):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predictions of shape (B,) or (B, 1)
            y_true: Ground truth of shape (B,) or (B, 1)

        Returns:
            loss: Scalar loss value (if reduction != 'none')
        """
        # Ensure both tensors have same shape
        if y_pred.dim() != y_true.dim():
            if y_pred.dim() == 2 and y_pred.size(1) == 1:
                y_pred = y_pred.squeeze(1)
            if y_true.dim() == 2 and y_true.size(1) == 1:
                y_true = y_true.squeeze(1)

        # Calculate absolute error
        abs_error = torch.abs(y_true - y_pred)

        # Huber loss calculation
        quadratic = torch.min(
            abs_error, torch.tensor(self.delta, device=abs_error.device)
        )
        linear = abs_error - quadratic

        loss = 0.5 * quadratic.pow(2) + self.delta * linear

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    B, N = 4, 5
    v_pred = torch.randn(B, N, 3)
    v_true = torch.randn(B, N, 3)

    print(f"\nShould be the same:")
    loss1 = cosine_squared_loss(v_pred, v_true)
    print(f"cosine_squared_loss:                {loss1}")
    loss1_flipped = cosine_squared_loss(v_pred, -v_true)
    print(f"cosine_squared_loss (sign flipped): {loss1_flipped}")
    print(
        f"cosine_squared_loss (shape B, N*3): {cosine_squared_loss(v_pred.reshape(B, N*3), v_true.reshape(B, N*3))}"
    )

    print(f"\nShould be the same:")
    loss2 = L_ang_loss(v_pred, v_true)
    print(f"L_ang_loss:                {loss2}")
    loss2_flipped = L_ang_loss(v_pred, -v_true)
    print(f"L_ang_loss (sign flipped): {loss2_flipped}")
    print(
        f"L_ang_loss (shape B, N*3): {L_ang_loss(v_pred.reshape(B, N*3), v_true.reshape(B, N*3))}"
    )

    print(f"\nShould be the same:")
    loss3 = cosine_loss(v_pred, v_true)
    print(f"_cosine_loss:                {loss3}")
    loss3_flipped = cosine_loss(v_pred, -v_true)
    print(f"_cosine_loss (sign flipped): {loss3_flipped}")
    print(
        f"_cosine_loss (shape B, N*3): {cosine_loss(v_pred.reshape(B, N*3), v_true.reshape(B, N*3))}"
    )

    print(f"\nShould be the same:")
    loss4 = min_l2_loss(v_pred, v_true)
    print(f"_min_l2_loss:                {loss4}")
    loss4_flipped = min_l2_loss(v_pred, -v_true)
    print(f"_min_l2_loss (sign flipped): {loss4_flipped}")
    print(
        f"_min_l2_loss (shape B, N*3): {min_l2_loss(v_pred.reshape(B, N*3), v_true.reshape(B, N*3))}"
    )

    print(f"\nShould be the same:")
    loss5 = min_l1_loss(v_pred, v_true)
    print(f"_min_l1_loss:                {loss5}")
    loss5_flipped = min_l1_loss(v_pred, -v_true)
    print(f"_min_l1_loss (sign flipped): {loss5_flipped}")
    print(
        f"_min_l1_loss (shape B, N*3): {min_l1_loss(v_pred.reshape(B, N*3), v_true.reshape(B, N*3))}"
    )

    #####################################################################3
    # scalar losses

    # Example usage
    batch_size = 32

    # Test with different input shapes
    y_true_1d = torch.randn(batch_size)  # Shape: (B,)
    y_pred_1d = torch.randn(batch_size)  # Shape: (B,)

    y_true_2d = y_true_1d.unsqueeze(1)  # Shape: (B, 1)
    y_pred_2d = y_pred_1d.unsqueeze(1)  # Shape: (B, 1)

    # Initialize Huber loss
    huber_loss_fn = HuberLoss(delta=1.0, reduction="mean")

    print(f"\nComparison of scalar losses:")
    mse_loss_1d = F.mse_loss(y_pred_1d, y_true_1d)
    mse_loss_2d = F.mse_loss(y_pred_2d, y_true_2d)
    print(f"MSE Loss (1D): {mse_loss_1d.item():.4f}")
    print(f"MSE Loss (2D): {mse_loss_2d.item():.4f}")
    mae_loss_1d = F.l1_loss(y_pred_1d, y_true_1d)
    mae_loss_2d = F.l1_loss(y_pred_2d, y_true_2d)
    print(f"MAE Loss (1D): {mae_loss_1d.item():.4f}")
    print(f"MAE Loss (2D): {mae_loss_2d.item():.4f}")
    loss_1d = huber_loss_fn(y_pred_1d, y_true_1d)
    loss_2d = huber_loss_fn(y_pred_2d, y_true_2d)
    print(f"Huber Loss (δ=1.0) (1D): {loss_1d.item():.4f}")
    print(f"Huber Loss (δ=1.0) (2D): {loss_2d.item():.4f}")

    loss6 = log_mse_loss(y_pred_1d, y_true_1d)
    print(f"log_mse_loss (1D): {loss6.item():.4f}")
    loss6_flipped = log_mse_loss(y_pred_2d, y_true_2d)
    print(f"log_mse_loss (2D): {loss6_flipped.item():.4f}")
