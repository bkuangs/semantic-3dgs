"""GPU smoke tests for the parts of the pipeline that do not depend on your stubs."""

import json

import pytest
import torch
from conftest import cuda

pytestmark = [cuda, pytest.mark.cuda]


def test_rasterization_packs_rgb_semantics_and_depth():
    from gsplat import rasterization

    n, c = 50, 11
    means = torch.randn(n, 3, device="cuda") * 0.3 + torch.tensor([0, 0, 4.0], device="cuda")
    quats = torch.zeros(n, 4, device="cuda")
    quats[:, 0] = 1
    feats = torch.rand(n, 3 + c, device="cuda", requires_grad=True)
    K = torch.tensor([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]], device="cuda")
    out, alpha, info = rasterization(
        means, quats, torch.full((n, 3), 0.1, device="cuda"), torch.full((n,), 0.8, device="cuda"),
        feats, torch.eye(4, device="cuda")[None], K[None], 64, 48, render_mode="RGB+ED",
    )
    assert out.shape == (1, 48, 64, 3 + c + 1) and alpha.shape == (1, 48, 64, 1)
    depth = out[0, ..., -1][alpha[0, ..., 0] > 0.5]
    assert depth.numel() > 0 and (depth - 4.0).abs().max() < 1.5
    out[..., 3:3 + c].sum().backward()
    assert feats.grad[:, 3:].abs().sum() > 0


def test_synthetic_scene_generator(tmp_path):
    from semantic_3dgs.config import DataConfig
    from semantic_3dgs.data.scene import DrivingScene
    from semantic_3dgs.data.synthetic import generate_synthetic_scene

    root = generate_synthetic_scene(tmp_path / "synth", num_frames=3, width=96, height=64)
    meta = json.loads((root / "meta.json").read_text())
    assert len(meta["frames"]) == 3
    scene = DrivingScene(DataConfig(scene_dir=root, downscale=1))
    assert scene.num_images == 9
    img = scene.image(0)
    assert img.std() > 0.05, "render should not be blank"
    labels = scene.semantics(0)
    assert len(labels.unique()) >= 3
    assert {"cut_in_car", "parked_car", "pedestrian"} == set(scene.tracks)
