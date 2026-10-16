"""2D masks of tracked actors, obtained by projecting their 3D boxes."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.geometry import (
    box_corners,
    box_to_pose,
    project_points,
    transform_points,
    world_to_camera_viewmat,
)


def actor_mask(
    scene: DrivingScene,
    index: int,
    track_ids: Iterable[str],
    margin: float = 0.25,
    near: float = 0.1,
) -> Tensor:
    """(H, W) bool mask covering the projected 2D bounding rectangles of the given tracks.

    Conservative by design: corners behind the camera are clamped to the near plane, which
    stretches the rectangle to the image border instead of shrinking it.
    """
    rec = scene.images[index]
    cam = scene.cameras[rec.camera]
    mask = torch.zeros(cam.height, cam.width, dtype=torch.bool)
    viewmat = world_to_camera_viewmat(
        scene.image_T_world_vehicle(index), scene.T_vehicle_camera(rec.camera)
    )
    T_cam_vehicle = viewmat @ scene.T_world_vehicle(rec.frame)
    K = scene.K(rec.camera)
    for track_id in track_ids:
        box = scene.tracks[track_id].boxes.get(rec.frame)
        if box is None:
            continue
        T_vehicle_box = box_to_pose(
            torch.tensor(box.center, dtype=torch.float32),
            torch.tensor(box.heading, dtype=torch.float32),
        )
        size = torch.tensor(box.size, dtype=torch.float32) + 2 * margin
        corners = transform_points(T_cam_vehicle, box_corners(T_vehicle_box, size))
        if (corners[:, 2] <= near).all():
            continue
        corners = corners.clone()
        corners[:, 2] = corners[:, 2].clamp_min(near)
        uv, _ = project_points(corners, K)
        u0, v0 = uv.min(0).values.floor().long().tolist()
        u1, v1 = uv.max(0).values.ceil().long().tolist()
        u0, u1 = max(u0, 0), min(u1, cam.width)
        v0, v1 = max(v0, 0), min(v1, cam.height)
        if u1 > u0 and v1 > v0:
            mask[v0:v1, u0:u1] = True
    return mask
