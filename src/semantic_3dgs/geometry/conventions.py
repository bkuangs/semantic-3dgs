"""Coordinate-frame conventions. Milestone 0.

Frames used throughout the project
----------------------------------
* World ``W``: the vehicle frame at the first frame of the segment (preprocessing re-centres the
  Waymo global frame there so float32 stays precise). x forward, y left, z up.
* Vehicle ``V``: Waymo vehicle frame at a timestamp. x forward, y left, z up.
* Waymo camera ``C_w``: what Waymo calibration calls the camera frame. x forward (optical axis),
  y left, z up.
* OpenCV camera ``C_cv``: what gsplat expects. x right, y down, z forward (optical axis).

Notation: ``T_a_b`` maps coordinates expressed in frame ``b`` into frame ``a``:
``p_a = T_a_b @ p_b``. ``T_a_b @ T_b_c == T_a_c``.
"""

from __future__ import annotations

from torch import Tensor

from semantic_3dgs._stub import todo


def waymo_camera_from_opencv(dtype=None, device=None) -> Tensor:
    """Return ``T_cw_cv`` (4, 4): the rigid transform taking OpenCV-camera coordinates into
    Waymo-camera coordinates. Pure rotation, zero translation, determinant +1.
    """
    todo("M0", "write down where each OpenCV axis points in the Waymo camera frame")


def world_to_camera_viewmat(T_world_vehicle: Tensor, T_vehicle_camera: Tensor) -> Tensor:
    """OpenCV world-to-camera view matrix ``T_cv_W`` for gsplat.

    Args:
        T_world_vehicle: (..., 4, 4) ego pose at the image capture time.
        T_vehicle_camera: (..., 4, 4) Waymo camera extrinsic (camera -> vehicle, Waymo axes).

    Returns:
        (..., 4, 4) matrix mapping world points into the OpenCV camera frame.
    """
    todo("M0", "compose the chain W <- V <- C_w <- C_cv, then invert it")
