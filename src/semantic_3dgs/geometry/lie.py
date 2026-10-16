"""Lie-group utilities for rotations and rigid motions. Milestone 2/3.

Quaternions are ``(w, x, y, z)`` (gsplat convention). Twists are ``xi = (rho, phi)`` with
translational part ``rho`` first and rotational part ``phi`` second.
"""

from __future__ import annotations

from torch import Tensor

from semantic_3dgs._stub import todo


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product ``q1 ⊗ q2`` for (..., 4) wxyz quaternions (broadcasting)."""
    todo("M2")


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert (..., 3, 3) rotation matrices to unit wxyz quaternions with ``w >= 0``.

    Must be numerically robust for rotations near 180 degrees.
    """
    todo("M2", "Shepperd's method / branch on the largest diagonal term")


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert (..., 4) wxyz quaternions (not necessarily unit) to (..., 3, 3) rotations."""
    todo("M2")


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
