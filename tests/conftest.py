"""Shared fixtures: a tiny on-disk scene with exactly known geometry.

World layout (world = vehicle frame at frame 0; x forward, y left, z up):
  * the ego drives +1 m/frame in x and yaws +0.05 rad/frame;
  * ``mover``  (VEHICLE): centre (5 + 3f, 3, 1), heading 0.1 f, size (4, 2, 1) -> dynamic;
  * ``parked`` (VEHICLE): static in the world at (15, -3, 1), heading 0.3, size (4, 2, 1);
  * ``walker`` (PEDESTRIAN): static at (10, 5, 1), size (0.6, 0.6, 1);
  * a ground grid at z = 0 and a single "beacon" point at (11.5, 0, 1.5), which at frame 0 lies
    exactly 10 m down the optical axis of the FRONT camera (mounted at (1.5, 0, 1.5), no rotation).
Boxes and LiDAR are stored in each frame's *vehicle* frame, exactly like Waymo.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

NUM_FRAMES = 4
WIDTH, HEIGHT, FOCAL = 64, 48, 40.0


def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pose(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def box_grid(size, shrink=0.8, n=3) -> np.ndarray:
    axes = [np.linspace(-s / 2 * shrink, s / 2 * shrink, n) for s in size]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


@dataclass
class FakeScene:
    root: Path
    ego: list[np.ndarray]
    static_world: np.ndarray  # every non-mover point, world frame
    mover_local: np.ndarray  # mover points in its box frame
    mover_world: list[np.ndarray]  # T_W_O per frame


def mover_pose(f: int) -> np.ndarray:
    return pose(rz(0.1 * f), (5.0 + 3.0 * f, 3.0, 1.0))


PARKED_POSE = pose(rz(0.3), (15.0, -3.0, 1.0))
WALKER_POSE = pose(rz(0.0), (10.0, 5.0, 1.0))
BOX_SIZE = (4.0, 2.0, 1.0)
WALKER_SIZE = (0.6, 0.6, 1.0)
BEACON = np.array([[11.5, 0.0, 1.5]])


def write_fake_scene(root: Path) -> FakeScene:
    ego = [pose(rz(0.05 * f), (1.0 * f, 0.0, 0.0)) for f in range(NUM_FRAMES)]
    gx, gy = np.meshgrid(np.arange(-5.0, 26.0, 1.0), np.arange(-6.0, 7.0, 1.0))
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    parked = apply(PARKED_POSE, box_grid(BOX_SIZE))
    walker = apply(WALKER_POSE, box_grid(WALKER_SIZE))
    static_world = np.concatenate([ground, parked, walker, BEACON])
    mover_local = box_grid(BOX_SIZE)

    K = [[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]]
    T_vc = pose(np.eye(3), (1.5, 0.0, 1.5))
    meta = {
        "segment": "fake",
        "T_global_world": np.eye(4).tolist(),
        "cameras": {"FRONT": {"K": K, "width": WIDTH, "height": HEIGHT,
                              "T_vehicle_camera": T_vc.tolist(), "distortion": [0.0] * 5}},
        "frames": [],
    }
    tracks = {
        "mover": {"category": "VEHICLE", "boxes": {}},
        "parked": {"category": "VEHICLE", "boxes": {}},
        "walker": {"category": "PEDESTRIAN", "boxes": {}},
    }
    rng = np.random.default_rng(0)
    (root / "lidar").mkdir(parents=True)
    (root / "images" / "FRONT").mkdir(parents=True)
    for f in range(NUM_FRAMES):
        T_wv = ego[f]
        T_vw = np.linalg.inv(T_wv)
        world = np.concatenate([static_world, apply(mover_pose(f), mover_local)])
        np.savez(root / "lidar" / f"{f:04d}.npz", points=apply(T_vw, world).astype(np.float32))
        img = (rng.random((HEIGHT, WIDTH, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(root / "images" / "FRONT" / f"{f:04d}.png")
        meta["frames"].append({
            "index": f, "timestamp_us": f * 100_000, "T_world_vehicle": T_wv.tolist(),
            "lidar": f"lidar/{f:04d}.npz",
            "images": {"FRONT": {"path": f"images/FRONT/{f:04d}.png",
                                 "timestamp_us": f * 100_000, "T_world_vehicle": T_wv.tolist()}},
        })
        ego_yaw = 0.05 * f
        for name, T_wo, size in (("mover", mover_pose(f), BOX_SIZE),
                                 ("parked", PARKED_POSE, BOX_SIZE),
                                 ("walker", WALKER_POSE, WALKER_SIZE)):
            center_v = apply(T_vw, T_wo[:3, 3][None])[0]
            heading_w = math.atan2(T_wo[1, 0], T_wo[0, 0])
            tracks[name]["boxes"][str(f)] = {"center": center_v.tolist(), "size": list(size),
                                             "heading": heading_w - ego_yaw}
    sem = np.full((HEIGHT, WIDTH), 5, dtype=np.uint8)
    (root / "semantics" / "FRONT").mkdir(parents=True)
    Image.fromarray(sem).save(root / "semantics" / "FRONT" / "0000.png")
    (root / "meta.json").write_text(json.dumps(meta))
    (root / "tracks.json").write_text(json.dumps(tracks))
    return FakeScene(root, ego, static_world, mover_local, [mover_pose(f) for f in range(NUM_FRAMES)])


@pytest.fixture
def fake_scene(tmp_path) -> FakeScene:
    return write_fake_scene(tmp_path / "fake")


@pytest.fixture
def driving_scene(fake_scene):
    from semantic_3dgs.config import DataConfig
    from semantic_3dgs.data.scene import DrivingScene

    return DrivingScene(DataConfig(scene_dir=fake_scene.root, cameras=("FRONT",), downscale=1))


def ref_quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Independent reference (wxyz) used by tests only."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def ref_rz(yaw: float) -> torch.Tensor:
    return torch.tensor(rz(yaw), dtype=torch.float32)


def ref_pose(R, t) -> torch.Tensor:
    return torch.tensor(pose(np.asarray(R, dtype=np.float64), t), dtype=torch.float32)


def random_rotations(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    return ref_quat_to_rotmat(q)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
