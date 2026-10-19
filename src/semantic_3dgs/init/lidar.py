"""Background point cloud from LiDAR. Milestone 1."""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.geometry import box_to_pose, points_in_box, transform_points


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
    dynamic = []

    for tid, t in scene.tracks.items():
        if t.category not in categories:
            continue

        centers, poses = [], []
        for frame, box in t.boxes.items():
            T_wv = scene.T_world_vehicle(frame)     # ego -> world conversion to measure motion
            centers.append(torch.as_tensor(box.center, dtype=T_wv.dtype))
            poses.append(T_wv)

        # batched: row i is transformed by poses[i]
        centers_w = transform_points(torch.stack(poses), torch.stack(centers))  # (F, 3)

        if torch.cdist(centers_w, centers_w).max() > displacement_threshold:
            dynamic.append(tid)

    return sorted(dynamic)


def accumulate_static_points(
    scene: DrivingScene, excluded_track_ids: list[str], box_margin: float
) -> Tensor:
    """For every LiDAR sweep: drop the points that hit moving objects, 
    move the rest into the world frame, and concatenate everything into one cloud.

    Returns:
        (M, 3) float32 world-frame points (not yet downsampled).

    Useful API: ``scene.lidar(frame)`` (vehicle frame), ``scene.T_world_vehicle(frame)``,
    :func:`~semantic_3dgs.geometry.boxes.points_in_box`.
    """
    chunks = []
    for frame in range(scene.num_frames):
        pts = scene.lidar(frame)
        keep = torch.ones(len(pts), dtype=torch.bool)       # mask

        for tid in excluded_track_ids:                      # exclude dynamic tracks
            box = scene.tracks[tid].boxes.get(frame)
            if box is None: continue

            T_vb = box_to_pose(torch.as_tensor(box.center, dtype=torch.float32),    # box -> ego frame
                               torch.tensor(box.heading, dtype=torch.float32))
            size = torch.as_tensor(box.size, dtype=torch.float32)
            keep &= ~points_in_box(pts, T_vb, size, margin=box_margin)     # remove points inside moving tracks
            
        chunks.append(transform_points(scene.T_world_vehicle(frame), pts[keep]))

    return torch.cat(chunks).float()


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
