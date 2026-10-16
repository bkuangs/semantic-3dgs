"""Milestone 0: frame conventions, rigid transforms, projection."""

import math

import torch
from conftest import HEIGHT, WIDTH, ref_pose, ref_rz

from semantic_3dgs.geometry import (
    invert_se3,
    project_points,
    transform_points,
    waymo_camera_from_opencv,
    world_to_camera_viewmat,
)


def test_waymo_camera_from_opencv_axes():
    T = waymo_camera_from_opencv()
    assert T.shape == (4, 4)
    R = T[:3, :3]
    # OpenCV optical axis (+z) is Waymo camera +x; OpenCV right (+x) is Waymo -y; down is -z.
    torch.testing.assert_close(R @ torch.tensor([0.0, 0, 1]), torch.tensor([1.0, 0, 0]))
    torch.testing.assert_close(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, -1, 0]))
    torch.testing.assert_close(R @ torch.tensor([0.0, 1, 0]), torch.tensor([0.0, 0, -1]))
    torch.testing.assert_close(torch.linalg.det(R), torch.tensor(1.0))
    torch.testing.assert_close(T[:3, 3], torch.zeros(3))


def test_viewmat_identity_mount():
    viewmat = world_to_camera_viewmat(torch.eye(4), torch.eye(4))
    p = transform_points(viewmat, torch.tensor([[10.0, 2.0, 1.0]]))  # ahead, left, up
    torch.testing.assert_close(p, torch.tensor([[-2.0, -1.0, 10.0]]))


def test_viewmat_with_ego_pose_and_mount():
    T_world_vehicle = ref_pose(ref_rz(math.pi / 2), (5.0, 0.0, 0.0))  # facing world +y
    T_vehicle_camera = ref_pose(torch.eye(3), (1.0, 0.0, 1.5))
    viewmat = world_to_camera_viewmat(T_world_vehicle, T_vehicle_camera)
    p = transform_points(viewmat, torch.tensor([[5.0, 10.0, 1.5]]))
    torch.testing.assert_close(p, torch.tensor([[0.0, 0.0, 9.0]]), atol=1e-5, rtol=0)


def test_viewmat_is_batched():
    T_wv = torch.eye(4).repeat(5, 1, 1)
    T_wv[:, 0, 3] = torch.arange(5.0)
    out = world_to_camera_viewmat(T_wv, torch.eye(4).expand(5, 4, 4))
    assert out.shape == (5, 4, 4)
    torch.testing.assert_close(out[:, 2, 3], -torch.arange(5.0))


def test_invert_se3_matches_inverse():
    R = ref_rz(0.7) @ torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    T = ref_pose(R, (1.0, -2.0, 3.0)).repeat(3, 1, 1)
    torch.testing.assert_close(invert_se3(T), torch.linalg.inv(T), atol=1e-6, rtol=0)
    torch.testing.assert_close(invert_se3(T) @ T, torch.eye(4).expand(3, 4, 4), atol=1e-6, rtol=0)


def test_transform_points_single_and_batched():
    T = ref_pose(ref_rz(math.pi / 2), (1.0, 0.0, 0.0))
    p = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
    torch.testing.assert_close(
        transform_points(T, p), torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]), atol=1e-6, rtol=0
    )
    Ts = torch.stack([torch.eye(4), T])
    out = transform_points(Ts, p)
    torch.testing.assert_close(out[0], p[0])
    torch.testing.assert_close(out[1], torch.tensor([0.0, 0.0, 2.0]), atol=1e-6, rtol=0)


def test_project_points():
    K = torch.tensor([[100.0, 0, 50], [0, 80.0, 40], [0, 0, 1]])
    uv, depth = project_points(torch.tensor([[0.0, 0.0, 2.0], [1.0, -1.0, 4.0]]), K)
    torch.testing.assert_close(uv, torch.tensor([[50.0, 40.0], [75.0, 20.0]]))
    torch.testing.assert_close(depth, torch.tensor([2.0, 4.0]))


def test_lidar_depth_image_hits_the_beacon(driving_scene):
    from semantic_3dgs.data.depth import lidar_depth_image

    depth = lidar_depth_image(driving_scene, 0)
    assert depth.shape == (HEIGHT, WIDTH)
    cx, cy = int(WIDTH / 2), int(HEIGHT / 2)
    torch.testing.assert_close(depth[cy, cx], torch.tensor(10.0), atol=1e-4, rtol=0)
    assert (depth > 0).sum() > 20
