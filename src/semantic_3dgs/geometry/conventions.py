"""Coordinate-frame conventions. Milestone 0.

Frames used throughout the project
----------------------------------
* World ``W``: the vehicle frame at the first frame of the segment (preprocessing re-centres the
  dataset's global frame there so float32 stays precise). x forward, y left, z up.
* Vehicle ``V``: ego vehicle frame at a timestamp. x forward, y left, z up.
* FLU camera ``C_flu``: the camera frame stored in ``T_vehicle_camera``, with the same
  forward-left-up axes as the vehicle: x forward (optical axis), y left, z up.
* OpenCV camera ``C_cv``: what gsplat expects. x right, y down, z forward (optical axis).

Notation: ``T_a_b`` maps coordinates expressed in frame ``b`` into frame ``a``:
``p_a = T_a_b @ p_b``. ``T_a_b @ T_b_c == T_a_c``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs._stub import todo
from semantic_3dgs.geometry.transforms import invert_se3


def flu_camera_from_opencv(dtype=None, device=None) -> Tensor:
    """Build rotation matrix for OpenCV-camera -> FLU-camera coordinates."""
    return torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        dtype=dtype if dtype is not None else torch.float32,
        device=device,
    )


def world_to_camera_viewmat(T_world_vehicle: Tensor, T_vehicle_camera: Tensor) -> Tensor:
    """Build the OpenCV world-to-camera view matrix for gsplat; only the extrinsic part of M [R | t].

    Args:
        T_world_vehicle: (..., 4, 4) ego pose at the image capture time.
        T_vehicle_camera: (..., 4, 4) camera extrinsic (camera -> vehicle, FLU camera axes).

    Returns:
        (..., 4, 4) matrix mapping world points into the OpenCV camera frame.
    """
    T_W_Ccv = T_world_vehicle @ T_vehicle_camera @ flu_camera_from_opencv(
        dtype=T_world_vehicle.dtype, device=T_world_vehicle.device)
    
    return invert_se3(T_W_Ccv)
