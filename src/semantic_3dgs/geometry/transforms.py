"""Rigid transforms and pinhole projection. Milestone 0."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def invert_se3(T: Tensor) -> Tensor:
    """
    Invert rigid transforms (..., 4, 4) using the SE(3) structure, not ``torch.inverse``.
    
    TODO: Learn why this is a better option than torch.linalg.inv()
    """

    R = T[..., :3, :3]    # 3x3
    t = T[..., :3, 3:]    # 3x1
    # For an orthonormal matrix, inverse = transpose
    R_t = R.transpose(-1, -2)   # NOT R @ t
    
    out = torch.zeros_like(T)
    out[..., :3, :3] = R_t
    out[..., :3, 3:] = -R_t @ t
    out[..., 3, 3] = 1.0          # homogeneous coordinate

    return out


def transform_points(T: Tensor, points: Tensor) -> Tensor:
    """Apply a rigid transform.

    Args:
        T: (4, 4) or (N, 4, 4) transforms.
        points: (N, 3) points.

    Returns:
        (N, 3) transformed points. When ``T`` is batched, row ``i`` uses ``T[i]``.
    """
    p_h = F.pad(points, (0, 1), value=1.0)                  # turn into homogeneous coordinate

    # T is 4x4 but p_h is still Nx4, so we unsqueeze to make it 4x1
    # Use .squeeze() afterwards to remove the dim
    return (T @ p_h.unsqueeze(-1)).squeeze(-1)[..., :3]     # index [:3] to undo the padding


def project_points(points_cam: Tensor, K: Tensor) -> tuple[Tensor, Tensor]:
    """Pinhole projection of OpenCV-camera-frame points.

    Args:
        points_cam: (N, 3) points in the OpenCV camera frame.
        K: (3, 3) intrinsics.

    Returns:
        uv: (N, 2) pixel coordinates ``(K @ p / z)[:2]`` (x right, y down).
        depth: (N,) z coordinate. Points behind the camera keep their (negative) depth; callers
            filter on ``depth > 0``.
    """
    uvw = points_cam @ K.T      # same as K @ p
    depth = points_cam[:, 2]
    uv = uvw[:, :2] / uvw[:, 2:3]

    return uv, depth


def pixel_ray_directions(K: Tensor, T_world_cv: Tensor, height: int, width: int) -> Tensor:
    """For every pixel, compute the world-space direction the camera is looking through that
    pixel. These are used to color the sky.

    This is needed since the sky is infinitely far away, so the color only depends on direction
    rather than where the camera is. Gaussians handle this poorly; use a sky cubemap instead, which
    computes a learned color for every direction.

    Wherever the Gaussians don't fully cover a pixel (alpha < 1), the sky color for that pixel's 
    direction shows through.

    Args:
        K: (3, 3) intrinsics.
        T_world_cv: (4, 4) OpenCV camera-to-world transform.
        height, width: image size.

    Returns:
        (height, width, 3) unit vectors. Pixel (row v, col u) uses image point (u + 0.5, v + 0.5).
    """
    dtype, device = K.dtype, K.device
    v, u = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device) + 0.5,
        torch.arange(width, dtype=dtype, device=device) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1)     # (H, W, 3) homogeneous pixels
    dirs_cam = pixels @ torch.linalg.inv(K).T                     # K^-1 @ [u, v, 1] per pixel
    dirs_world = dirs_cam @ T_world_cv[:3, :3].to(dtype).T        # rotate only: directions ignore t
    return F.normalize(dirs_world, dim=-1)
