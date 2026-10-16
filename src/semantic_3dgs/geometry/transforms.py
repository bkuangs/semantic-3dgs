"""Rigid transforms and pinhole projection. Milestone 0."""

from __future__ import annotations

from torch import Tensor

from semantic_3dgs._stub import todo


def invert_se3(T: Tensor) -> Tensor:
    """Invert rigid transforms (..., 4, 4) using the SE(3) structure, not ``torch.inverse``."""
    todo("M0", "[R | t]^-1 = [R^T | -R^T t]")


def transform_points(T: Tensor, points: Tensor) -> Tensor:
    """Apply a rigid transform.

    Args:
        T: (4, 4) or (N, 4, 4) transforms.
        points: (N, 3) points.

    Returns:
        (N, 3) transformed points. When ``T`` is batched, row ``i`` uses ``T[i]``.
    """
    todo("M0")


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
    todo("M0")


def pixel_ray_directions(K: Tensor, T_world_cv: Tensor, height: int, width: int) -> Tensor:
    """Unit ray directions in the world frame through every pixel centre.

    Args:
        K: (3, 3) intrinsics.
        T_world_cv: (4, 4) OpenCV camera-to-world transform.
        height, width: image size.

    Returns:
        (height, width, 3) unit vectors. Pixel (row v, col u) uses image point (u + 0.5, v + 0.5).
    """
    todo("M3", "needed by the sky model")
