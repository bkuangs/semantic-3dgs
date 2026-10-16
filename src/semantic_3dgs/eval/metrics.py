"""Image, semantic and depth metrics (all implemented)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    """PSNR in dB for (H, W, 3) images in [0, 1], optionally over a (H, W) mask."""
    err = (pred.clamp(0, 1) - target) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        err = err[mask]
    return float(-10.0 * torch.log10(err.mean().clamp_min(1e-10)))


def _gaussian_window(size: int, sigma: float, channels: int, device) -> Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim_map(pred: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Per-pixel SSIM (H, W) between (H, W, 3) images (differentiable)."""
    x = pred.permute(2, 0, 1)[None]
    y = target.permute(2, 0, 1)[None]
    channels = x.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, x.device)
    pad = window_size // 2

    def filt(z: Tensor) -> Tensor:
        return F.conv2d(F.pad(z, [pad] * 4, mode="replicate"), window, groups=channels)

    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x**2
    syy = filt(y * y) - mu_y**2
    sxy = filt(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean(1)[0]


def ssim(pred: Tensor, target: Tensor) -> float:
    return float(ssim_map(pred.clamp(0, 1), target).mean())


class LPIPS:
    """Lazy LPIPS wrapper (``pip install semantic-3dgs[evaluation]``)."""

    def __init__(self, device):
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor) -> float:
        x = pred.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1
        y = target.permute(2, 0, 1)[None] * 2 - 1
        return float(self.net(x, y))


def confusion_matrix(pred: Tensor, target: Tensor, num_classes: int, ignore_index: int) -> Tensor:
    valid = target != ignore_index
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes**2).view(num_classes, num_classes).cpu()


def miou_from_confusion(conf: Tensor) -> tuple[float, Tensor]:
    """Mean IoU over classes present in the ground truth, plus per-class IoU (NaN if absent)."""
    conf = conf.double()
    tp = conf.diag()
    denom = conf.sum(0) + conf.sum(1) - tp
    iou = torch.where(conf.sum(1) > 0, tp / denom.clamp_min(1), torch.full_like(tp, float("nan")))
    return float(iou[~iou.isnan()].mean()) if (~iou.isnan()).any() else float("nan"), iou


def depth_errors(pred: Tensor, lidar: Tensor, mask: Tensor | None = None) -> dict[str, float]:
    valid = lidar > 0
    if mask is not None:
        valid &= mask
    if valid.sum() == 0:
        return {"depth_abs_rel": float("nan"), "depth_rmse": float("nan")}
    p, g = pred[valid], lidar[valid]
    return {
        "depth_abs_rel": float(((p - g).abs() / g).mean()),
        "depth_rmse": float(((p - g) ** 2).mean().sqrt()),
    }
