"""Oriented 3D bounding boxes. Milestone 1/2.

Waymo boxes are 7-DoF: centre ``(x, y, z)``, size ``(length, width, height)`` along the box's
own ``(x, y, z)`` axes, and ``heading`` = yaw about +z of the parent frame (radians, CCW).
The box frame ``O`` has its origin at the box centre.
"""

from __future__ import annotations

from torch import Tensor

from semantic_3dgs._stub import todo


def box_to_pose(center: Tensor, heading: Tensor) -> Tensor:
    """Pose ``T_parent_O`` of box frames.

    Args:
        center: (..., 3) box centres in the parent frame.
        heading: (...,) yaw angles.

    Returns:
        (..., 4, 4) transforms from box-local coordinates into the parent frame.
    """
    todo("M1")


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
    todo("M1")


def box_corners(T_parent_box: Tensor, size: Tensor) -> Tensor:
    """(8, 3) corner coordinates in the parent frame (any fixed corner order)."""
    todo("M1")
