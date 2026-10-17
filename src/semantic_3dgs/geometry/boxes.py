"""Oriented 3D bounding boxes. Milestone 1/2.

Waymo boxes are 7-DoF: centre ``(x, y, z)``, size ``(length, width, height)`` along the box's
own ``(x, y, z)`` axes, and ``heading`` = yaw about +z of the parent frame (radians, CCW).
The box frame ``O`` has its origin at the box centre.
"""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs._stub import todo
from semantic_3dgs.geometry.transforms import invert_se3, transform_points


def box_to_pose(center: Tensor, heading: Tensor) -> Tensor:
    """Pose ``T_parent_O`` of box frames.

    Args:
        center: (..., 3) box centres in the parent frame.
        heading: (...,) yaw angles.

    Returns:
        (..., 4, 4) transforms from box-local coordinates into the parent frame.
    """
    c, s = torch.cos(heading), torch.sin(heading)
    T = torch.zeros(*heading.shape, 4, 4, dtype=center.dtype, device=center.device)
    T[..., 0, 0] = c
    T[..., 0, 1] = -s
    T[..., 1, 0] = s
    T[..., 1, 1] = c
    T[..., :3, 3] = center
    T[..., 2, 2] = 1.0
    T[..., 3, 3] = 1.0

    return T


def points_in_box(points: Tensor, T_parent_box: Tensor, size: Tensor, margin: float = 0.0) -> Tensor:
    """Boolean mask of points inside an oriented box.

    Args:
        points: (N, 3) points in the parent frame.
        T_parent_box: (4, 4) box pose.
        size: (3,) full extents ``(length, width, height)``.
        margin: metres added to *each* half extent (boxes are tight; LiDAR returns on mirrors,
            wheels and the ground contact patch sit just outside them).

    Returns:
        (N,) bool mask.
    """
    p_local = transform_points(invert_se3(T_parent_box), points)    # (N, 3) in box frame
    half = size / 2 + margin                                        # (3,)
    return (p_local.abs() <= half).all(dim=-1)      


def box_corners(T_parent_box: Tensor, size: Tensor) -> Tensor:
    """(8, 3) corner coordinates in the parent frame (any fixed corner order)."""
    l, w, h = size

    points = [
        [l/2, w/2, h/2],
        [-l/2, w/2, h/2],
        [l/2, -w/2, h/2],
        [l/2, w/2, -h/2],
        [-l/2, -w/2, h/2],
        [l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2, w/2, -h/2],
    ]

    corners = torch.tensor(points, dtype=size.dtype, device=size.device)
    return transform_points(T_parent_box, corners)
