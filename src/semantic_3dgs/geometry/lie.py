"""Lie-group utilities for rotations and rigid motions. Milestone 2/3.

Quaternions are ``(w, x, y, z)`` (gsplat convention). Twists are ``xi = (rho, phi)`` with
translational part ``rho`` first and rotational part ``phi`` second.
"""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs._stub import todo


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """
    Hamilton product ``q1 ⊗ q2`` for (..., 4) wxyz quaternions (broadcasting).
    """
    w1, x1, y1, z1 = q1.unbind(-1)          # unbind(-1) splits off the last dimension
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert (..., 3, 3) rotation matrices to unit wxyz quaternions with ``w >= 0``.

    Must be numerically robust for rotations near 180 degrees.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # 4w², 4x², 4y², 4z²
    sq = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                      1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1).clamp_min(1e-12)

    # row i = 4·q_i·(w, x, y, z); dividing by 2·sqrt(sq_i) = 4|q_i| gives the quaternion
    cands = torch.stack([
        torch.stack([sq[..., 0], m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, sq[..., 1], m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, sq[..., 2], m12 + m21], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, sq[..., 3]], dim=-1),
    ], dim=-2) / (2 * sq.sqrt()[..., None])

    # pick the best-conditioned row
    idx = sq.argmax(dim=-1)[..., None, None].expand(*sq.shape[:-1], 1, 4)
    q = torch.gather(cands, -2, idx).squeeze(-2)

    # q and -q are the same rotation; choose w >= 0
    q = torch.where(q[..., :1] < 0, -q, q)
    return q / q.norm(dim=-1, keepdim=True)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """
    Convert a quaternion (not necessarily normalized) to 3x3 rotation matrix.
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1)

    return R.reshape(*q.shape[:-1], 3, 3)


def so3_exp(phi: Tensor) -> Tensor:
    """Rodrigues' formula: (..., 3) axis-angle -> (..., 3, 3) rotation.

    Must be finite *and differentiable* at ``phi == 0`` (use a Taylor expansion there).
    """
    todo("M3")


def se3_exp(xi: Tensor) -> Tensor:
    """Exponential map (..., 6) twist ``(rho, phi)`` -> (..., 4, 4) rigid transform.

    Translation is ``V(phi) @ rho`` with the SO(3) left Jacobian ``V``. Must be differentiable at
    zero, since pose corrections are initialised there.
    """
    todo("M3")


def se3_log(T: Tensor) -> Tensor:
    """Logarithm map (..., 4, 4) -> (..., 6) twist ``(rho, phi)``; inverse of :func:`se3_exp`."""
    todo("M3")
