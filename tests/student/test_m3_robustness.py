"""Milestone 3: Lie-group pose refinement, sky, exposure, depth & sky losses."""

import math

import torch
from conftest import random_rotations, ref_pose, ref_rz

from semantic_3dgs.geometry.lie import se3_exp, se3_log, so3_exp
from semantic_3dgs.geometry.transforms import pixel_ray_directions
from semantic_3dgs.losses import depth_loss, sky_loss
from semantic_3dgs.model.exposure import ExposureCorrection
from semantic_3dgs.model.pose_refine import PoseRefinement
from semantic_3dgs.model.sky import SkyCubemap


def test_so3_exp():
    torch.testing.assert_close(so3_exp(torch.zeros(3)), torch.eye(3))
    torch.testing.assert_close(so3_exp(torch.tensor([0.0, 0.0, 0.3])), ref_rz(0.3), atol=1e-6,
                               rtol=0)
    R = so3_exp(torch.randn(10, 3))
    torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(3).expand(10, 3, 3), atol=1e-5,
                               rtol=0)


def test_exp_maps_are_differentiable_at_zero():
    phi = torch.zeros(3, requires_grad=True)
    so3_exp(phi)[0, 1].backward()
    assert torch.isfinite(phi.grad).all() and phi.grad.abs().sum() > 0
    xi = torch.zeros(6, requires_grad=True)
    se3_exp(xi)[:3, :].sum().backward()
    assert torch.isfinite(xi.grad).all()


def test_se3_exp_translation_and_rotation():
    T = se3_exp(torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(T, ref_pose(torch.eye(3), (1.0, 2.0, 3.0)))
    T = se3_exp(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]))
    torch.testing.assert_close(T, ref_pose(ref_rz(math.pi / 2), (0, 0, 0)), atol=1e-6, rtol=0)


def test_se3_log_inverts_exp():
    g = torch.Generator().manual_seed(0)
    xi = torch.randn(16, 6, generator=g) * torch.tensor([2.0, 2.0, 2.0, 0.5, 0.5, 0.5])
    torch.testing.assert_close(se3_log(se3_exp(xi)), xi, atol=1e-4, rtol=0)


def test_pose_refinement_identity_at_init_and_background_fixed():
    refine = PoseRefinement(num_nodes=3, num_frames=5)
    base = torch.stack([torch.eye(4), ref_pose(ref_rz(0.4), (1, 2, 3)),
                        ref_pose(random_rotations(1)[0], (0, 0, 1))])
    torch.testing.assert_close(refine(base, 2), base, atol=1e-6, rtol=0)
    with torch.no_grad():
        refine.deltas.normal_()
    torch.testing.assert_close(refine(base, 2)[0], base[0])


def test_pose_refinement_acts_in_object_frame():
    refine = PoseRefinement(num_nodes=2, num_frames=1)
    with torch.no_grad():
        refine.deltas[1, 0] = torch.tensor([1.0, 0, 0, 0, 0, 0])  # 1 m "forward" for the car
    base = torch.stack([torch.eye(4), ref_pose(ref_rz(math.pi / 2), (5.0, 0.0, 0.0))])
    out = refine(base, 0)
    torch.testing.assert_close(out[1, :3, 3], torch.tensor([5.0, 1.0, 0.0]), atol=1e-6, rtol=0)


def test_pose_refinement_smoothness():
    refine = PoseRefinement(num_nodes=2, num_frames=6)
    torch.testing.assert_close(refine.smoothness_loss(), torch.tensor(0.0))
    with torch.no_grad():
        refine.deltas[1, ::2, 0] = 0.1
    loss = refine.smoothness_loss()
    assert loss > 0
    loss.backward()
    assert refine.deltas.grad[1].abs().sum() > 0


def test_pixel_ray_directions():
    K = torch.tensor([[2.0, 0, 2.5], [0, 2.0, 1.5], [0, 0, 1]])
    rays = pixel_ray_directions(K, torch.eye(4), height=3, width=5)
    assert rays.shape == (3, 5, 3)
    torch.testing.assert_close(rays.norm(dim=-1), torch.ones(3, 5))
    torch.testing.assert_close(rays[1, 2], torch.tensor([0.0, 0.0, 1.0]))
    assert rays[0, 0, 0] < 0 and rays[0, 0, 1] < 0
    T = ref_pose(ref_rz(0.5), (10.0, 0.0, 0.0))
    rotated = pixel_ray_directions(K, T, height=3, width=5)
    torch.testing.assert_close(rotated, rays @ ref_rz(0.5).T, atol=1e-6, rtol=0)


def test_sky_cubemap_direction_lookup():
    sky = SkyCubemap(resolution=8)
    with torch.no_grad():
        sky.faces.fill_(-10.0)
        sky.faces[0, ..., 0] = 10.0  # +x face is red
    dirs = torch.tensor([[1.0, 0.1, -0.2], [5.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rgb = sky(dirs)
    assert rgb.shape == (4, 3)
    assert ((rgb >= 0) & (rgb <= 1)).all()
    assert rgb[0, 0] > 0.99 and rgb[1, 0] > 0.99
    assert rgb[2, 0] < 0.01 and rgb[3, 0] < 0.01


def test_sky_cubemap_shapes_and_gradients():
    sky = SkyCubemap(resolution=4)
    rgb = sky(torch.randn(2, 6, 7, 3))
    assert rgb.shape == (2, 6, 7, 3)
    torch.testing.assert_close(rgb, torch.full_like(rgb, 0.5))
    rgb.sum().backward()
    assert sky.faces.grad.abs().sum() > 0


def test_exposure_correction():
    exp = ExposureCorrection(num_images=3)
    img = torch.rand(4, 5, 3)
    torch.testing.assert_close(exp(img, 1), img)
    torch.testing.assert_close(exp.regularization(), torch.tensor(0.0))
    with torch.no_grad():
        exp.affine[2, :, :3] *= 2.0
        exp.affine[2, :, 3] = 0.1
    torch.testing.assert_close(exp(img, 2), img * 2 + 0.1)
    assert exp.regularization() > 0
    exp(img, 2).sum().backward()
    assert exp.affine.grad[2].abs().sum() > 0 and exp.affine.grad[0].abs().sum() == 0


def test_depth_loss():
    pred = torch.full((4, 4), 5.0, requires_grad=True)
    lidar = torch.zeros(4, 4)
    lidar[0, 0], lidar[1, 1], lidar[2, 2] = 6.0, 4.0, 100.0
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[2, 2] = False
    loss = depth_loss(pred, lidar, mask)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert pred.grad[0, 1] == 0 and pred.grad[2, 2] == 0


def test_sky_loss():
    sky = torch.zeros(4, 4, dtype=torch.bool)
    sky[:2] = True
    low = torch.zeros(4, 4)
    low[2:] = 1.0  # opaque ground is fine
    high = low.clone()
    high[:2] = 0.9
    assert sky_loss(low, sky) < 1e-4
    assert sky_loss(high, sky) > sky_loss(high * 0.5, sky) > sky_loss(low, sky)
    other = low.clone()
    other[2:] = 0.3
    torch.testing.assert_close(sky_loss(other, sky), sky_loss(low, sky))
