"""Milestone 2: quaternion algebra, scene-graph composition, object point clouds, SH frames."""

import math

import torch
from conftest import NUM_FRAMES, random_rotations, ref_pose, ref_quat_to_rotmat, ref_rz

from semantic_3dgs.geometry.lie import quat_multiply, quat_to_rotmat, rotmat_to_quat
from semantic_3dgs.init.objects import accumulate_object_points
from semantic_3dgs.model.scene_graph import build_node_tables, compose_gaussians
from semantic_3dgs.model.sh import object_frame_view_dirs

S = math.sqrt(0.5)


def test_quat_multiply_composes_rotations():
    qz90 = torch.tensor([S, 0.0, 0.0, S])
    q = quat_multiply(qz90, qz90)
    torch.testing.assert_close(ref_quat_to_rotmat(q), ref_rz(math.pi), atol=1e-6, rtol=0)
    qx90 = torch.tensor([S, S, 0.0, 0.0])
    torch.testing.assert_close(
        ref_quat_to_rotmat(quat_multiply(qz90, qx90)),
        ref_quat_to_rotmat(qz90) @ ref_quat_to_rotmat(qx90), atol=1e-6, rtol=0,
    )


def test_quat_multiply_broadcasts():
    out = quat_multiply(torch.tensor([1.0, 0, 0, 0]), torch.randn(7, 4))
    assert out.shape == (7, 4)


def test_quat_to_rotmat_matches_reference():
    q = torch.randn(32, 4)
    torch.testing.assert_close(quat_to_rotmat(q), ref_quat_to_rotmat(q), atol=1e-5, rtol=0)


def test_rotmat_to_quat_roundtrip_including_near_pi():
    R = torch.cat([random_rotations(64), ref_rz(math.pi - 1e-4)[None], ref_rz(math.pi)[None]])
    q = rotmat_to_quat(R)
    assert (q[:, 0] >= 0).all()
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(len(R)), atol=1e-5, rtol=0)
    torch.testing.assert_close(ref_quat_to_rotmat(q), R, atol=1e-4, rtol=0)


def _two_node_scene():
    means = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [S, 0.0, 0.0, S]])
    node_ids = torch.tensor([0, 1, 1])
    poses = torch.stack([torch.eye(4), ref_pose(ref_rz(math.pi / 2), (10.0, 0.0, 0.0))])
    return means, quats, node_ids, poses


def test_compose_places_objects_and_leaves_background():
    means, quats, node_ids, poses = _two_node_scene()
    out = compose_gaussians(means, quats, node_ids, poses, torch.tensor([True, False]))
    torch.testing.assert_close(out.means, torch.tensor(
        [[1.0, 0.0, 0.0], [10.0, 1.0, 0.0], [8.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    assert out.visible.tolist() == [True, False, False]
    torch.testing.assert_close(out.rotations[0], torch.eye(3))
    torch.testing.assert_close(out.rotations[2], ref_rz(math.pi / 2), atol=1e-6, rtol=0)


def test_compose_rotates_covariances():
    means, quats, node_ids, poses = _two_node_scene()
    out = compose_gaussians(means, quats, node_ids, poses, torch.ones(2, dtype=torch.bool))
    R_world = ref_quat_to_rotmat(out.quats)
    R_expected = out.rotations @ ref_quat_to_rotmat(quats)
    torch.testing.assert_close(R_world, R_expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(R_world[2], ref_rz(math.pi), atol=1e-6, rtol=0)


def test_compose_is_differentiable_wrt_poses():
    means, quats, node_ids, poses = _two_node_scene()
    poses = poses.clone().requires_grad_(True)
    out = compose_gaussians(means, quats, node_ids, poses, torch.ones(2, dtype=torch.bool))
    out.means.sum().backward()
    assert poses.grad[1].abs().sum() > 0


def test_node_tables_follow_world_trajectory(driving_scene, fake_scene):
    poses, visible = build_node_tables(driving_scene, [None, "mover"])
    assert poses.shape == (2, NUM_FRAMES, 4, 4)
    assert visible.all()
    torch.testing.assert_close(poses[0], torch.eye(4).expand(NUM_FRAMES, 4, 4))
    expected = torch.tensor(fake_scene.mover_world, dtype=torch.float32)
    torch.testing.assert_close(poses[1], expected, atol=1e-5, rtol=0)


def test_object_points_are_canonical(driving_scene, fake_scene):
    pts = accumulate_object_points(driving_scene, "mover", box_margin=0.25)
    local = torch.tensor(fake_scene.mover_local, dtype=torch.float32)
    assert pts.shape == (len(local) * NUM_FRAMES, 3)
    assert torch.cdist(pts.double(), local.double()).min(1).values.max() < 1e-3


def test_view_dirs_background_matches_standard_3dgs():
    means = torch.tensor([[0.0, 0.0, 5.0], [3.0, 4.0, 0.0]])
    rot = torch.eye(3).expand(2, 3, 3)
    dirs = object_frame_view_dirs(means, torch.zeros(3), rot)
    torch.testing.assert_close(dirs, torch.tensor([[0.0, 0.0, 1.0], [0.6, 0.8, 0.0]]))


def test_view_dirs_are_invariant_to_rigidly_moving_object_and_camera():
    R = random_rotations(1, seed=3)[0]
    t = torch.tensor([4.0, -1.0, 2.0])
    mean_local = torch.tensor([[1.0, 2.0, 0.5]])
    cam_local = torch.tensor([-3.0, 0.0, 1.0])
    d0 = object_frame_view_dirs(mean_local, cam_local, torch.eye(3)[None])
    d1 = object_frame_view_dirs(mean_local @ R.T + t, R @ cam_local + t, R[None])
    torch.testing.assert_close(d0, d1, atol=1e-5, rtol=0)
    torch.testing.assert_close(d1.norm(dim=-1), torch.ones(1))
