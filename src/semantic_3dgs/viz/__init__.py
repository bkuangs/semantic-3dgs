"""Visualisation helpers: semantic palettes, depth colormaps, panels and videos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from semantic_3dgs.classes import IGNORE_INDEX, PALETTE, SKY

# Coarse samples of the Turbo colormap (Google, Apache-2.0), interpolated linearly.
_TURBO = np.array(
    [
        [48, 18, 59], [70, 107, 227], [41, 187, 236], [43, 239, 160], [134, 254, 70],
        [218, 226, 55], [254, 165, 49], [229, 79, 14], [122, 4, 3],
    ],
    dtype=np.float32,
)


def to_uint8(image: Tensor) -> np.ndarray:
    return (image.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def colorize_labels(labels: Tensor | np.ndarray) -> np.ndarray:
    labels = labels.cpu().numpy() if isinstance(labels, Tensor) else labels
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels != IGNORE_INDEX
    out[valid] = PALETTE[labels[valid]]
    return out


def semantic_prediction(logits: Tensor, alpha: Tensor, sky_threshold: float = 0.5) -> Tensor:
    """argmax labels, with pixels the Gaussians do not cover assigned to sky."""
    labels = logits.argmax(-1)
    labels[alpha < sky_threshold] = SKY
    return labels


def colorize_depth(depth: Tensor, max_depth: float = 80.0, valid: Tensor | None = None) -> np.ndarray:
    d = (depth.detach().cpu().numpy() / max_depth).clip(0, 1)
    x = np.linspace(0, 1, len(_TURBO))
    rgb = np.stack([np.interp(d, x, _TURBO[:, c]) for c in range(3)], -1).astype(np.uint8)
    if valid is not None:
        rgb[~valid.cpu().numpy()] = 0
    return rgb


def hstack(*images: np.ndarray) -> np.ndarray:
    return np.concatenate(images, axis=1)


def save_png(image: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def write_video(frames: list[np.ndarray], path: str | Path, fps: int = 10) -> None:
    import imageio.v2 as imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # yuv420p needs even dimensions.
    h, w = frames[0].shape[:2]
    frames = [f[: h - h % 2, : w - w % 2] for f in frames]
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def overlay_points(image: Tensor, depth: Tensor, max_depth: float = 60.0, radius: int = 1) -> np.ndarray:
    """Draw sparse depth (H, W, 0 = empty) on top of an image."""
    base = to_uint8(image).copy()
    colors = colorize_depth(depth, max_depth)
    ys, xs = torch.nonzero(depth > 0, as_tuple=True)
    h, w = depth.shape
    for dy in range(-radius + 1, radius):
        for dx in range(-radius + 1, radius):
            yy = (ys + dy).clamp(0, h - 1).numpy()
            xx = (xs + dx).clamp(0, w - 1).numpy()
            base[yy, xx] = colors[ys.numpy(), xs.numpy()]
    return base


def overlay_mask(image: np.ndarray, mask: Tensor, color=(255, 0, 255), strength: float = 0.4):
    out = image.astype(np.float32)
    m = mask.cpu().numpy()
    out[m] = (1 - strength) * out[m] + strength * np.asarray(color, dtype=np.float32)
    return out.astype(np.uint8)
