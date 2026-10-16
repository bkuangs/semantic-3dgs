"""Sparse LiDAR depth images for depth supervision and evaluation."""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.geometry import project_points, transform_points, world_to_camera_viewmat


def lidar_depth_image(
    scene: DrivingScene, index: int, near: float = 0.5, occlusion_window: int = 7,
    occlusion_ratio: float = 0.1,
) -> Tensor:
    """(H, W) float32 depth along the optical axis from the LiDAR sweep of the image's frame.

    Pixels without a return are 0. When several points land on one pixel the nearest wins.

    The LiDAR sits above the cameras, so it sees surfaces the camera cannot (e.g. the road behind
    a car), and with sparse returns those far points leak through the gaps between the near
    ones. A return is dropped if it is more than ``occlusion_ratio`` farther than the nearest
    return within ``occlusion_window`` pixels (set the window to 0 to disable).
    Dynamic objects are slightly misaligned because LiDAR and camera are not captured at the
    same instant - mask them if that bites.
    """
    rec = scene.images[index]
    cam = scene.cameras[rec.camera]
    viewmat = world_to_camera_viewmat(
        scene.image_T_world_vehicle(index), scene.T_vehicle_camera(rec.camera)
    )
    T_cam_lidar_vehicle = viewmat @ scene.T_world_vehicle(rec.frame)
    points_cam = transform_points(T_cam_lidar_vehicle, scene.lidar(rec.frame))
    uv, depth = project_points(points_cam, scene.K(rec.camera))

    u = torch.floor(uv[:, 0]).long()
    v = torch.floor(uv[:, 1]).long()
    keep = (depth > near) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    flat = v[keep] * cam.width + u[keep]

    out = torch.full((cam.height * cam.width,), float("inf"))
    out.scatter_reduce_(0, flat, depth[keep].float(), reduce="amin")
    out = out.view(cam.height, cam.width)
    if occlusion_window > 1:
        local_min = -torch.nn.functional.max_pool2d(
            -out[None, None], occlusion_window, stride=1, padding=occlusion_window // 2
        )[0, 0]
        out[out > local_min * (1 + occlusion_ratio)] = float("inf")
    out[torch.isinf(out)] = 0.0
    return out
