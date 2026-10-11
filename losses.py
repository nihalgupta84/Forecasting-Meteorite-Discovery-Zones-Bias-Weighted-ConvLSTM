"""
losses.py
─────────
Custom loss functions for spatiotemporal meteorite density map prediction.

Combined loss:
    L = λ1 * MSE(pred, target)
      + λ2 * (1 - SpatialIoU(pred>τ, target>τ))
      + λ3 * BiasWeightedMSE(pred, target, mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


class SpatialIoULoss(nn.Module):
    """
    Differentiable spatial IoU loss over 2D density maps.
    Uses soft binarisation (sigmoid) to allow gradient flow.

    IoU = intersection / union  (on soft masks)
    Loss = 1 - IoU
    """

    def __init__(self, threshold: float = config.IOU_THRESHOLD,
                 smooth: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.smooth    = smooth

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, 1, H, W) in [0, 1]
            target : (B, 1, H, W) in [0, 1]
        Returns:
            scalar loss
        """
        # Soft binarisation: steepen sigmoid around threshold
        pred_bin   = torch.sigmoid(10 * (pred   - self.threshold))
        target_bin = torch.sigmoid(10 * (target - self.threshold))

        intersection = (pred_bin * target_bin).sum(dim=[1, 2, 3])
        union        = (pred_bin + target_bin - pred_bin * target_bin
                        ).sum(dim=[1, 2, 3])

        iou  = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou
        return loss.mean()


class BiasWeightedMSE(nn.Module):
    """
    MSE loss weighted by the discovery bias mask.

    Down-weights Antarctic / Saharan zones where collection programs
    inflate the training signal. Unbiased zones get full weight.
    """

    def __init__(self, bias_mask: torch.Tensor):
        """
        Args:
            bias_mask : (H, W) float tensor with weights in [0, 1]
                        (loaded from bias_mask.npy via build_bias_mask())
        """
        super().__init__()
        # Register as buffer (moves with .to(device) but not a parameter)
        self.register_buffer("mask", bias_mask.unsqueeze(0).unsqueeze(0))
                                     # → (1, 1, H, W)

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, 1, H, W)
            target : (B, 1, H, W)
        Returns:
            scalar weighted MSE
        """
        diff = (pred - target) ** 2                  # (B, 1, H, W)
        weighted = diff * self.mask                  # broadcast over batch
        return weighted.mean()


class CombinedLoss(nn.Module):
    """
    Full combined loss = λ1*MSE + λ2*IoULoss + λ3*BiasWeightedMSE

    Args:
        bias_mask  : (H, W) numpy array or tensor
        lambda_mse : weight for standard MSE
        lambda_iou : weight for spatial IoU loss
        lambda_bias: weight for bias-weighted MSE
    """

    def __init__(self,
                 bias_mask,
                 lambda_mse:  float = config.LAMBDA_MSE,
                 lambda_iou:  float = config.LAMBDA_IOU,
                 lambda_bias: float = config.LAMBDA_BIAS_MSE):
        super().__init__()

        self.lambda_mse  = lambda_mse
        self.lambda_iou  = lambda_iou
        self.lambda_bias = lambda_bias

        # Convert mask to tensor if numpy
        if isinstance(bias_mask, np.ndarray):
            bias_mask = torch.from_numpy(bias_mask.astype(np.float32))

        self.mse_loss  = nn.MSELoss()
        self.iou_loss  = SpatialIoULoss()
        self.bias_loss = BiasWeightedMSE(bias_mask)

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> tuple:
        """
        Returns:
            total_loss : scalar
            components : dict with individual loss values (for logging)
        """
        mse  = self.mse_loss(pred, target)
        iou  = self.iou_loss(pred, target)
        bias = self.bias_loss(pred, target)

        total = (self.lambda_mse  * mse +
                 self.lambda_iou  * iou +
                 self.lambda_bias * bias)

        components = {
            "loss_total": total.item(),
            "loss_mse":   mse.item(),
            "loss_iou":   iou.item(),
            "loss_bias":  bias.item(),
        }
        return total, components


# ── Evaluation metrics (non-differentiable, for reporting) ────────────────────

def compute_metrics(pred:   torch.Tensor,
                    target: torch.Tensor,
                    threshold: float = config.IOU_THRESHOLD) -> dict:
    """
    Compute MSE, SSIM, and Spatial IoU on a batch.
    For use during validation / test (no gradients needed).

    Args:
        pred   : (B, 1, H, W)
        target : (B, 1, H, W)
    Returns:
        dict of scalar metrics
    """
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(pred, target).item()

        # Spatial IoU (hard binarisation at threshold)
        pred_bin   = (pred   > threshold).float()
        target_bin = (target > threshold).float()
        intersection = (pred_bin * target_bin).sum(dim=[1, 2, 3])
        union        = (pred_bin + target_bin).clamp(max=1.0).sum(dim=[1, 2, 3])
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()

        # SSIM (simplified 2D, per-image)
        ssim_val = _batch_ssim(pred, target)

        # MAE
        mae = (pred - target).abs().mean().item()

    return {
        "mse":  mse,
        "mae":  mae,
        "iou":  iou,
        "ssim": ssim_val,
    }


def _batch_ssim(pred: torch.Tensor, target: torch.Tensor,
                window_size: int = 11, C1: float = 0.01**2,
                C2: float = 0.03**2) -> float:
    """
    Simplified SSIM averaged over batch.
    Uses uniform window (convolution) rather than Gaussian for speed.
    """
    B = pred.shape[0]
    pred_   = pred.view(B, -1)
    target_ = target.view(B, -1)

    mu_x  = pred_.mean(dim=1, keepdim=True)
    mu_y  = target_.mean(dim=1, keepdim=True)
    var_x = ((pred_  - mu_x)**2).mean(dim=1)
    var_y = ((target_ - mu_y)**2).mean(dim=1)
    cov   = ((pred_ - mu_x) * (target_ - mu_y)).mean(dim=1)

    num   = (2 * mu_x.squeeze() * mu_y.squeeze() + C1) * (2 * cov + C2)
    denom = (mu_x.squeeze()**2 + mu_y.squeeze()**2 + C1) * (var_x + var_y + C2)

    ssim_per_img = num / (denom + 1e-8)
    return ssim_per_img.mean().item()


if __name__ == "__main__":
    import numpy as np
    from kde_maps import build_bias_mask

    mask = build_bias_mask()
    loss_fn = CombinedLoss(mask)

    # Dummy tensors
    pred   = torch.rand(2, 1, config.GRID_H, config.GRID_W)
    target = torch.rand(2, 1, config.GRID_H, config.GRID_W)

    total, comps = loss_fn(pred, target)
    print("Combined loss test:")
    for k, v in comps.items():
        print(f"  {k}: {v:.4f}")

    metrics = compute_metrics(pred, target)
    print("\nMetrics test:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
