"""View-dependent colour for Gaussians that live in moving frames. Milestone 2."""

from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F

from semantic_3dgs._stub import todo


def object_frame_view_dirs(
    means_world: Tensor, camera_center_world: Tensor, rotations: Tensor
) -> Tensor:
    """Each Gaussian's color depends on the direction you look at it from, via spherical-harmonics (SH) 
    coefficients. We query Gaussians with a viewing direction to get RGB color.

    For moving objects, the SH coefficients are learned in the car's own frame. Thus, we must express
    the viewing direction in the car's frame before evaluating SH.

    This function simply computes an object's local frame for each Gaussian, depending on the camera's
    viewing direction.

    Args:
        means_world: (N, 3) world positions.
        camera_center_world: (3,) camera centre in the world.
        rotations: (N, 3, 3) ``R_W_node`` per Gaussian (from :class:`ComposedGaussians`).

    Returns:
        (N, 3) unit vectors. For the background (identity rotation) this equals the standard
        3DGS direction ``normalize(mu - c)``.
    """
    d_W = F.normalize(means_world - camera_center_world, dim=-1)    # world direction from camera to Gaussian

    # Now "from car's right" always means the car's own right side
    return (rotations.transpose(-1, -2) @ d_W.unsqueeze(-1)).squeeze(-1)  # rotate into node frame


def evaluate_sh(sh_coeffs: Tensor, dirs: Tensor, degree: int) -> Tensor:
    """RGB from SH coefficients (N, K, 3) and unit directions (N, 3), using gsplat's kernel and
    the standard 3DGS ``+0.5`` offset."""
    from gsplat import spherical_harmonics

    return (spherical_harmonics(degree, dirs, sh_coeffs) + 0.5).clamp_min(0.0)
