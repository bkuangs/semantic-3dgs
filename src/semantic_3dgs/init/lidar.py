"""Background point cloud from LiDAR. Milestone 1."""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs._stub import todo
from semantic_3dgs.data.scene import DrivingScene


def classify_dynamic_tracks(
    scene: DrivingScene, categories: tuple[str, ...], displacement_threshold: float
) -> list[str]:
    """Return the sorted ids of tracks that should become dynamic scene-graph nodes.

    A track qualifies if its category is in ``categories`` and its box centre, expressed in the
    *world* frame, moves more than ``displacement_threshold`` metres over its lifetime (parked
    cars are boxed too, but they belong to the background).

    Useful API: ``scene.tracks``, ``track.boxes[frame].center / .heading``,
    ``scene.T_world_vehicle(frame)``.
    """
    todo("M1")


def accumulate_static_points(
    scene: DrivingScene, excluded_track_ids: list[str], box_margin: float
) -> Tensor:
    """Concatenate every LiDAR sweep in the world frame after removing points that fall inside
    the (margin-inflated) boxes of ``excluded_track_ids`` at that frame.

    Returns:
        (M, 3) float32 world-frame points (not yet downsampled).

    Useful API: ``scene.lidar(frame)`` (vehicle frame), ``scene.T_world_vehicle(frame)``,
    :func:`~semantic_3dgs.geometry.boxes.points_in_box`.
    """
    todo("M1")


def voxel_downsample(points: Tensor, voxel_size: float) -> Tensor:
    """Average all points that fall into the same ``voxel_size`` cube."""
    if points.numel() == 0 or voxel_size <= 0:
        return points
    keys = torch.floor(points / voxel_size).long()
    keys -= keys.min(0).values
    dims = keys.max(0).values + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    unique, inverse = torch.unique(flat, return_inverse=True)
    sums = torch.zeros(len(unique), 3, dtype=points.dtype).index_add_(0, inverse, points)
    counts = torch.zeros(len(unique), dtype=points.dtype).index_add_(
        0, inverse, torch.ones_like(points[:, 0])
    )
    return sums / counts[:, None]


def random_subsample(points: Tensor, max_points: int, seed: int = 0) -> Tensor:
    if len(points) <= max_points:
        return points
    generator = torch.Generator().manual_seed(seed)
    return points[torch.randperm(len(points), generator=generator)[:max_points]]
