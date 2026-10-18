"""Lie-group utilities for rotations and rigid motions. Milestone 2/3.

Quaternions are ``(w, x, y, z)`` (gsplat convention). Twists are ``xi = (rho, phi)`` with
translational part ``rho`` first and rotational part ``phi`` second.
"""

from __future__ import annotations

import torch
from torch import Tensor


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
    """Problem: We want to learn a rotation with gradient descent, but its 9 entries (R) must
    stay orthonormal with det=1. A gradient step breaks that.

    Solution: Learn a free vector phi instead, then map it to a valid rotation with R = exp(φ).

    Terms:
    * SO(3) is the set of all valid 3D rotation matrices.
    * so(3) (lowercase) is the space of 3-vectors φ. 
        - The direction of φ is the rotation axis, and its length θ = |φ| is the angle in radians.
    * Exponential map exp: so(3) → SO(3) converts φ into R for that axis and angle.

    Use Rodrigues' formula: (..., 3) axis-angle -> (..., 3, 3) rotation.

    Must be finite *and differentiable* at ``phi == 0`` (use a Taylor expansion there).
    """
    K = _skew(phi)
    A, B, _ = _so3_coeffs(phi.norm(dim=-1, keepdim=True)[..., None])
    # Rodrigues'
    return torch.eye(3, dtype=phi.dtype, device=phi.device) + A * K + B * (K @ K)


def _skew(v: Tensor) -> Tensor:
    """(..., 3) -> (..., 3, 3) cross-product matrix, so that ``_skew(a) @ b == a x b``."""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    # Built with torch.stack: torch.Tensor([...]) would copy values and cut the gradient to v.
    return torch.stack([zero, -z, y,
                        z, zero, -x,
                        -y, x, zero], dim=-1).reshape(*v.shape[:-1], 3, 3)


def _so3_coeffs(theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``A = sin(θ)/θ``, ``B = (1 - cos θ)/θ²``, ``C = (θ - sin θ)/θ³`` for θ of shape (..., 1, 1).

    Each is 0/0 at θ = 0, so small angles use Taylor series. ``torch.where`` evaluates both
    branches, so the exact branch gets a safe θ to keep NaNs out of the gradients.
    """
    small = theta < 1e-2
    t = torch.where(small, torch.ones_like(theta), theta)
    A = torch.where(small, 1 - theta**2 / 6, torch.sin(t) / t)
    # Half-angle form: 1 - cos θ cancels catastrophically in float32 for small θ.
    B = torch.where(small, 0.5 - theta**2 / 24, 0.5 * (torch.sin(t / 2) / (t / 2)) ** 2)
    # θ - sin θ also cancels, so C switches to its series at a larger angle.
    small_c = theta < 1e-1
    tc = torch.where(small_c, torch.ones_like(theta), theta)
    C = torch.where(small_c, 1 / 6 - theta**2 / 120 + theta**4 / 5040, (tc - torch.sin(tc)) / tc**3)
    return A, B, C


def se3_exp(xi: Tensor) -> Tensor:
    """Convert 6 "learnable" numbers into a valid 4x4 rigid pose matrix:

            [ R   t ]      R = so3_exp(φ)
            [ 0   1 ]      t = V(φ) @ rho

    Input xi:
    * rho (first 3 numbers): translation
    * phi (last 3 numbers): rotation, same as in so3_exp

    Translation is ``V(phi) @ rho`` with the SO(3) left Jacobian ``V``. Must be differentiable at
    zero, since pose corrections are initialised there.
    """
    rho, phi = xi[..., :3], xi[..., 3:]
    K = _skew(phi)
    A, B, C = _so3_coeffs(phi.norm(dim=-1, keepdim=True)[..., None])
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)

    R = eye + A * K + B * (K @ K)                               # same as so3_exp(phi)
    V = eye + B * K + C * (K @ K)                               # SO(3) left Jacobian
    t = (V @ rho.unsqueeze(-1))                                 # (..., 3, 1)

    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=xi.dtype, device=xi.device)
    bottom = bottom.expand(*xi.shape[:-1], 1, 4)
    return torch.cat([torch.cat([R, t], dim=-1), bottom], dim=-2)


def so3_log(R: Tensor) -> Tensor:
    """(..., 3, 3) rotation -> (..., 3) axis-angle ``phi`` with ``|phi| <= pi``; inverse of so3_exp.

    Goes through the unit quaternion ``q = (cos θ/2, sin θ/2 · axis)``, which (unlike the trace
    formula) stays well conditioned near 180 degrees.
    """
    q = rotmat_to_quat(R)                                       # w >= 0, so θ in [0, π]
    w, v = q[..., :1], q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)                            # sin(θ/2)
    small = s < 1e-3
    s_safe = torch.where(small, torch.ones_like(s), s)
    # phi = θ · axis = (2·atan2(s, w) / s) · v; the factor tends to 2/w as s -> 0.
    factor = torch.where(small, 2 / w * (1 - s**2 / (3 * w**2)), 2 * torch.atan2(s_safe, w) / s_safe)
    return factor * v


def se3_log(T: Tensor) -> Tensor:
    """Logarithm map (..., 4, 4) -> (..., 6) twist ``(rho, phi)``

    The inverse of se3_exp(): ``phi = log(R)`` and ``rho = V(phi)^-1 @ t``.
    """
    phi = so3_log(T[..., :3, :3])
    theta = phi.norm(dim=-1, keepdim=True)[..., None]
    K = _skew(phi)
    # V^-1 = I - K/2 + D K², with D = (1 - (θ/2)·cot(θ/2)) / θ²  (-> 1/12 as θ -> 0)
    small = theta < 1e-1
    t = torch.where(small, torch.ones_like(theta), theta)
    D = torch.where(small, 1 / 12 + theta**2 / 720 + theta**4 / 30240,
                    (1 - (t / 2) / torch.tan(t / 2)) / t**2)
    V_inv = torch.eye(3, dtype=T.dtype, device=T.device) - 0.5 * K + D * (K @ K)
    rho = (V_inv @ T[..., :3, 3:]).squeeze(-1)
    return torch.cat([rho, phi], dim=-1)
