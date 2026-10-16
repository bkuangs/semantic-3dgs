"""Milestone 1: boxes, static/dynamic split, background initialisation, photometric loss."""

import math

import torch
from conftest import ref_pose, ref_rz

from semantic_3dgs.geometry import box_corners, box_to_pose, points_in_box
from semantic_3dgs.init.lidar import accumulate_static_points, classify_dynamic_tracks
from semantic_3dgs.losses import opacity_regularization, rgb_loss, scale_regularization


def test_box_to_pose():
    T = box_to_pose(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(math.pi / 2))
    expected = ref_pose(ref_rz(math.pi / 2), (1.0, 2.0, 3.0))
    torch.testing.assert_close(T, expected, atol=1e-6, rtol=0)


def test_box_to_pose_batched():
    T = box_to_pose(torch.zeros(2, 5, 3), torch.zeros(2, 5))
    assert T.shape == (2, 5, 4, 4)


def test_points_in_box_respects_heading_and_margin():
    T = box_to_pose(torch.tensor([10.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    size = torch.tensor([4.0, 2.0, 1.0])
    pts = torch.tensor([
        [10.0, 1.9, 1.0],   # inside: length now runs along world y
        [11.9, 0.0, 1.0],   # outside: width is only 2 m along world x
        [10.0, 2.1, 1.0],   # just outside, inside with margin
        [10.0, 0.0, 1.6],   # above the roof, inside with margin
    ])
    assert points_in_box(pts, T, size).tolist() == [True, False, False, False]
    assert points_in_box(pts, T, size, margin=0.2).tolist() == [True, False, True, True]


def test_box_corners_extent():
    T = box_to_pose(torch.tensor([0.0, 0.0, 0.0]), torch.tensor(0.0))
    corners = box_corners(T, torch.tensor([4.0, 2.0, 1.0]))
    assert corners.shape == (8, 3)
    torch.testing.assert_close(corners.max(0).values, torch.tensor([2.0, 1.0, 0.5]))
    torch.testing.assert_close(corners.min(0).values, torch.tensor([-2.0, -1.0, -0.5]))


def test_only_world_moving_vehicles_are_dynamic(driving_scene):
    # "parked" moves 3 m in the *vehicle* frame (the ego drives past it) but not in the world.
    assert classify_dynamic_tracks(driving_scene, ("VEHICLE",), 1.0) == ["mover"]
    assert classify_dynamic_tracks(driving_scene, ("PEDESTRIAN",), 1.0) == []
    assert classify_dynamic_tracks(driving_scene, ("VEHICLE",), 100.0) == []


def test_static_points_are_in_world_frame_and_exclude_movers(driving_scene, fake_scene):
    pts = accumulate_static_points(driving_scene, ["mover"], box_margin=0.25)
    static = torch.tensor(fake_scene.static_world, dtype=torch.float32)
    assert pts.shape == (len(static) * len(fake_scene.ego), 3)
    nearest = torch.cdist(pts.double(), static.double()).min(1).values
    assert nearest.max() < 1e-3, "points must land on the world-frame static geometry"


def test_rgb_loss_l1_and_mask():
    pred = torch.zeros(4, 4, 3, requires_grad=True)
    target = torch.zeros(4, 4, 3)
    target[:2] = 1.0
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[2:] = True
    loss = rgb_loss(pred, target, mask, ssim_lambda=0.0)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    mask[:] = True
    loss = rgb_loss(pred, target, mask, ssim_lambda=0.0)
    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_rgb_loss_masked_pixels_get_no_gradient():
    pred = torch.rand(8, 8, 3, requires_grad=True)
    target = torch.rand(8, 8, 3)
    mask = torch.ones(8, 8, dtype=torch.bool)
    mask[:, :4] = False
    rgb_loss(pred, target, mask, ssim_lambda=0.2).backward()
    assert pred.grad[:, :4].abs().max() == 0
    assert pred.grad[:, 4:].abs().max() > 0


def test_rgb_loss_with_ssim_is_zero_for_identical_images():
    img = torch.rand(16, 16, 3)
    loss = rgb_loss(img, img, torch.ones(16, 16, dtype=torch.bool), ssim_lambda=0.2)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=0)


def test_regularizers():
    torch.testing.assert_close(opacity_regularization(torch.tensor([0.0, 0.5, 1.0])),
                               torch.tensor(0.5))
    torch.testing.assert_close(scale_regularization(torch.full((3, 3), 0.2)), torch.tensor(0.2))
