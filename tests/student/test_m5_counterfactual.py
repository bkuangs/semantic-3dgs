"""Milestone 5: counterfactual trajectory edits."""

import math

import torch
from conftest import ref_pose, ref_rz

from semantic_3dgs.counterfactual import offset_pose


def test_zero_offset_is_identity():
    T = ref_pose(ref_rz(0.3), (1.0, 2.0, 3.0))
    torch.testing.assert_close(offset_pose(T), T)


def test_offset_is_expressed_in_local_axes():
    T = ref_pose(ref_rz(math.pi / 2), (5.0, 0.0, 0.0))  # facing world +y
    out = offset_pose(T, left=2.0)
    torch.testing.assert_close(out[:3, 3], torch.tensor([3.0, 0.0, 0.0]), atol=1e-6, rtol=0)
    out = offset_pose(T, forward=1.0, up=0.5)
    torch.testing.assert_close(out[:3, 3], torch.tensor([5.0, 1.0, 0.5]), atol=1e-6, rtol=0)
    torch.testing.assert_close(out[:3, :3], T[:3, :3])


def test_yaw_rotates_about_local_z():
    T = ref_pose(ref_rz(0.2), (0.0, 0.0, 0.0))
    out = offset_pose(T, yaw=0.3)
    torch.testing.assert_close(out[:3, :3], ref_rz(0.5), atol=1e-6, rtol=0)
