"""View-dependent colour for Gaussians that live in moving frames. Milestone 2."""

from __future__ import annotations

from torch import Tensor

from semantic_3dgs._stub import todo


def object_frame_view_dirs(
    means_world: Tensor, camera_center_world: Tensor, rotations: Tensor
) -> Tensor:
    """Unit viewing directions expressed in each Gaussian's *node-local* frame.

    SH coefficients of an object are learned in its canonical frame, so they must be evaluated
    with the direction "camera -> Gaussian" rotated into that frame. Otherwise a highlight on a
    car stays fixed in world space while the car drives past it.

    Args:
        means_world: (N, 3) world positions.
        camera_center_world: (3,) camera centre in the world.
        rotations: (N, 3, 3) ``R_W_node`` per Gaussian (from :class:`ComposedGaussians`).

    Returns:
        (N, 3) unit vectors. For the background (identity rotation) this equals the standard
        3DGS direction ``normalize(mu - c)``.
    """
    todo("M2")


def evaluate_sh(sh_coeffs: Tensor, dirs: Tensor, degree: int) -> Tensor:
    """RGB from SH coefficients (N, K, 3) and unit directions (N, 3), using gsplat's kernel and
    the standard 3DGS ``+0.5`` offset."""
    from gsplat import spherical_harmonics

    return (spherical_harmonics(degree, dirs, sh_coeffs) + 0.5).clamp_min(0.0)
