"""Procedural driving scene written in the preprocessed format, rendered with gsplat.

Use it to debug every milestone in minutes before touching Waymo data. It contains a static
street, a parked car (static track), a car that cuts into the ego lane (dynamic track), and a
pedestrian (non-rigid actor -> masked), seen by three cameras with different exposure gains.

Spoiler warning: this file necessarily contains ground-truth frame conventions and box poses.
Implement Milestone 0/1 from the docs and tests before reading it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from semantic_3dgs.classes import CLASS_INDEX, NUM_CLASSES, SKY

_T_CW_CV = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

CAMERAS = {
    # name: (translation in vehicle frame, yaw deg, exposure gain)
    "FRONT": ((1.5, 0.0, 1.6), 0.0, 1.0),
    "FRONT_LEFT": ((1.3, 0.6, 1.6), 50.0, 0.85),
    "FRONT_RIGHT": ((1.3, -0.6, 1.6), -50.0, 1.15),
}


def _rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _pose(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


@dataclass
class _Cloud:
    points: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    scale: np.ndarray

    @staticmethod
    def cat(clouds: list[_Cloud]) -> _Cloud:
        return _Cloud(*(np.concatenate([getattr(c, f) for c in clouds]) for f in
                        ("points", "colors", "labels", "scale")))


def _grid(origin, u, v, extent_u, extent_v, spacing, rng) -> np.ndarray:
    nu, nv = max(int(extent_u / spacing), 1), max(int(extent_v / spacing), 1)
    a, b = np.meshgrid((np.arange(nu) + 0.5) / nu * extent_u, (np.arange(nv) + 0.5) / nv * extent_v)
    pts = np.asarray(origin) + a.reshape(-1, 1) * np.asarray(u) + b.reshape(-1, 1) * np.asarray(v)
    return pts + rng.normal(0, spacing * 0.15, pts.shape)


def _surface(points, color, label, spacing, rng, noise=0.04) -> _Cloud:
    colors = np.clip(np.asarray(color) + rng.normal(0, noise, (len(points), 3)), 0, 1)
    return _Cloud(points, colors, np.full(len(points), CLASS_INDEX[label]),
                  np.full(len(points), spacing * 0.7))


def _static_world(rng) -> _Cloud:
    parts: list[_Cloud] = []
    x0, length = -15.0, 110.0
    road = _grid((x0, -7, 0), (1, 0, 0), (0, 1, 0), length, 14, 0.12, rng)
    patch = 0.06 * np.sin(road[:, 0] * 0.7) * np.cos(road[:, 1] * 1.3)
    cloud = _surface(road, (0.33, 0.33, 0.35), "road", 0.12, rng)
    cloud.colors = np.clip(cloud.colors + patch[:, None], 0, 1)
    parts.append(cloud)
    for start in np.arange(x0, x0 + length, 9.0):  # dashed centre line
        pts = _grid((start, -0.08, 0.01), (1, 0, 0), (0, 1, 0), 3.0, 0.16, 0.04, rng)
        parts.append(_surface(pts, (0.95, 0.95, 0.9), "lane_marking", 0.04, rng, 0.02))
    for y in (-6.9, 6.75):  # solid edge lines
        pts = _grid((x0, y, 0.01), (1, 0, 0), (0, 1, 0), length, 0.15, 0.05, rng)
        parts.append(_surface(pts, (0.95, 0.95, 0.9), "lane_marking", 0.05, rng, 0.02))
    for y0 in (7.0, -10.0):
        pts = _grid((x0, y0, 0.15), (1, 0, 0), (0, 1, 0), length, 3.0, 0.12, rng)
        parts.append(_surface(pts, (0.72, 0.68, 0.6), "sidewalk", 0.12, rng))
    palette = [(0.7, 0.45, 0.35), (0.55, 0.6, 0.7), (0.8, 0.75, 0.55), (0.5, 0.4, 0.45)]
    for side, y in ((1, 10.0), (-1, -10.0)):
        for i, bx in enumerate(np.arange(x0, x0 + length, 12.0)):
            pts = _grid((bx, y, 0), (1, 0, 0), (0, 0, 1), 12.0, 9.0 + 3 * ((i + side) % 3), 0.12, rng)
            base = np.asarray(palette[(i + (side > 0)) % len(palette)])
            cloud = _surface(pts, base, "building", 0.12, rng)
            windows = ((pts[:, 0] - bx) % 3.0 > 1.0) & ((pts[:, 2] % 3.0) > 1.2)
            cloud.colors[windows] = 0.18 + rng.normal(0, 0.02, (windows.sum(), 3))
            parts.append(cloud)
    for x in np.arange(5.0, 90.0, 17.0):  # trees on alternating sidewalks
        y = 8.5 if int(x) % 2 else -8.5
        d = rng.normal(size=(1500, 3))
        crown = np.array([x, y, 4.5]) + 1.6 * d / np.linalg.norm(d, axis=1, keepdims=True)
        parts.append(_surface(crown, (0.2, 0.5, 0.15), "vegetation", 0.15, rng, 0.06))
        trunk = _grid((x - 0.15, y, 0.15), (1, 0, 0), (0, 0, 1), 0.3, 3.0, 0.05, rng)
        parts.append(_surface(trunk, (0.35, 0.25, 0.15), "vegetation", 0.05, rng))
    for x in (18.0, 55.0):  # signs
        pole = _grid((x, -7.5, 0.15), (1, 0, 0), (0, 0, 1), 0.1, 2.6, 0.03, rng)
        parts.append(_surface(pole, (0.6, 0.6, 0.6), "traffic_infra", 0.03, rng, 0.01))
        plate = _grid((x, -7.9, 2.4), (0, 1, 0), (0, 0, 1), 0.8, 0.6, 0.03, rng)
        parts.append(_surface(plate, (0.8, 0.1, 0.1), "traffic_infra", 0.03, rng, 0.01))
    return _Cloud.cat(parts)


def _car_local(size, body, rng, spacing=0.06) -> _Cloud:
    ln, w, h = size
    faces = [
        ((-ln / 2, -w / 2, -h / 2), (1, 0, 0), (0, 1, 0), ln, w),
        ((-ln / 2, -w / 2, h / 2), (1, 0, 0), (0, 1, 0), ln, w),
        ((-ln / 2, -w / 2, -h / 2), (1, 0, 0), (0, 0, 1), ln, h),
        ((-ln / 2, w / 2, -h / 2), (1, 0, 0), (0, 0, 1), ln, h),
        ((-ln / 2, -w / 2, -h / 2), (0, 1, 0), (0, 0, 1), w, h),
        ((ln / 2, -w / 2, -h / 2), (0, 1, 0), (0, 0, 1), w, h),
    ]
    pts = np.concatenate([_grid(o, u, v, eu, ev, spacing, rng) for o, u, v, eu, ev in faces])
    cloud = _surface(pts, body, "vehicle", spacing, rng, 0.02)
    glass = pts[:, 2] > h * 0.1
    cloud.colors[glass] = (0.12, 0.14, 0.18)
    front = pts[:, 0] > ln / 2 - 0.02
    rear = pts[:, 0] < -ln / 2 + 0.02
    lamps = np.abs(np.abs(pts[:, 1]) - w * 0.35) < 0.15
    lamps &= np.abs(pts[:, 2] + h * 0.15) < 0.1
    cloud.colors[front & lamps] = (1.0, 1.0, 0.9)
    cloud.colors[rear & lamps] = (0.9, 0.05, 0.05)
    return cloud


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _render(means, colors, scales, viewmat, K, width, height, device):
    from gsplat import rasterization

    n = len(means)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1
    out, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales[:, None].expand(n, 3).contiguous(),
        opacities=torch.full((n,), 0.97, device=device), colors=colors,
        viewmats=viewmat[None], Ks=K[None], width=width, height=height,
        near_plane=0.1, far_plane=500.0,
    )
    return out[0], alpha[0, ..., 0]


def generate_synthetic_scene(
    out_dir: str | Path,
    num_frames: int = 40,
    width: int = 480,
    height: int = 320,
    seed: int = 0,
    device: str = "cuda",
) -> Path:
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    dt = 0.1
    static = _static_world(rng)
    K = np.array([[260.0, 0, width / 2], [0, 260.0, height / 2], [0, 0, 1]])

    ego = [_pose(np.eye(3), (6.0 * i * dt, 0.0, 0.0)) for i in range(num_frames)]

    actors = {
        "parked_car": ("VEHICLE", (4.6, 2.0, 1.6), _car_local((4.6, 2.0, 1.6), (0.75, 0.1, 0.1), rng),
                       lambda t: ((30.0, -5.0, 0.8), 0.05)),
        "cut_in_car": ("VEHICLE", (4.4, 1.9, 1.5), _car_local((4.4, 1.9, 1.5), (0.1, 0.25, 0.8), rng),
                       None),
        "pedestrian": ("PEDESTRIAN", (0.6, 0.6, 1.75), _car_local((0.6, 0.6, 1.75), (0.9, 0.6, 0.2), rng, 0.04),
                       lambda t: ((20.0 - 1.4 * t, 8.5, 1.05), np.pi)),
    }

    def cut_in(t):
        x = 8.0 + 9.0 * t
        y = 3.5 - 3.5 * _smoothstep((t - 1.2) / 2.0)
        ds = 1e-3
        y2 = 3.5 - 3.5 * _smoothstep((t + ds - 1.2) / 2.0)
        return (x, y, 0.75), float(np.arctan2((y2 - y) / ds, 9.0))

    actors["cut_in_car"] = (*actors["cut_in_car"][:3], cut_in)

    cams = {}
    for name, (t, yaw, gain) in CAMERAS.items():
        R = _rz(np.deg2rad(yaw)) @ _ry(np.deg2rad(3.0))
        cams[name] = {"T": _pose(R, t), "gain": gain}

    dev = torch.device(device)
    K_t = torch.tensor(K, dtype=torch.float32, device=dev)
    sky_rows = torch.linspace(0, 1, height, device=dev)[:, None, None]
    sky = (1 - sky_rows) * torch.tensor([0.35, 0.55, 0.85], device=dev) + sky_rows * torch.tensor(
        [0.75, 0.82, 0.9], device=dev)

    meta = {"segment": "synthetic", "T_global_world": np.eye(4).tolist(), "cameras": {}, "frames": []}
    for name, cam in cams.items():
        meta["cameras"][name] = {"K": K.tolist(), "width": width, "height": height,
                                 "T_vehicle_camera": cam["T"].tolist(), "distortion": [0.0] * 5}
    tracks = {tid: {"category": cat, "boxes": {}} for tid, (cat, *_rest) in actors.items()}

    for f in range(num_frames):
        t = f * dt
        T_wv = ego[f]
        T_vw = np.linalg.inv(T_wv)
        clouds = [static]
        for tid, (_cat, size, local, traj) in actors.items():
            center, heading = traj(t)
            T_wo = _pose(_rz(heading), center)
            moved = _Cloud(_apply(T_wo, local.points), local.colors, local.labels, local.scale)
            clouds.append(moved)
            c_v = _apply(T_vw, np.asarray(center)[None])[0]
            tracks[tid]["boxes"][str(f)] = {"center": c_v.tolist(), "size": list(size),
                                            "heading": float(heading)}
        world = _Cloud.cat(clouds)

        near = np.linalg.norm(world.points[:, :2] - T_wv[:2, 3], axis=1) < 50.0
        idx = rng.choice(np.nonzero(near)[0], size=min(40_000, near.sum()), replace=False)
        lidar = _apply(T_vw, world.points[idx] + rng.normal(0, 0.01, (len(idx), 3)))
        (out / "lidar").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "lidar" / f"{f:04d}.npz", points=lidar.astype(np.float32))

        means = torch.tensor(world.points, dtype=torch.float32, device=dev)
        onehot = torch.nn.functional.one_hot(torch.tensor(world.labels, device=dev), NUM_CLASSES)
        feats = torch.cat([torch.tensor(world.colors, dtype=torch.float32, device=dev),
                           onehot.float()], 1)
        scales = torch.tensor(world.scale, dtype=torch.float32, device=dev)

        entry = {"index": f, "timestamp_us": int(t * 1e6), "T_world_vehicle": T_wv.tolist(),
                 "lidar": f"lidar/{f:04d}.npz", "images": {}}
        for name, cam in cams.items():
            T_w_cv = T_wv @ cam["T"] @ _T_CW_CV
            viewmat = torch.tensor(np.linalg.inv(T_w_cv), dtype=torch.float32, device=dev)
            img, alpha = _render(means, feats, scales, viewmat, K_t, width, height, dev)
            rgb = img[..., :3] + (1 - alpha)[..., None] * sky
            rgb = (rgb * cam["gain"]).clamp(0, 1)
            labels = img[..., 3:].argmax(-1)
            labels[alpha < 0.5] = SKY
            img_path = out / "images" / name / f"{f:04d}.jpg"
            sem_path = out / "semantics" / name / f"{f:04d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            sem_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_u8 = (rgb.cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb_u8).save(img_path, quality=95)
            Image.fromarray(labels.cpu().numpy().astype(np.uint8)).save(sem_path)
            entry["images"][name] = {"path": f"images/{name}/{f:04d}.jpg",
                                     "timestamp_us": entry["timestamp_us"],
                                     "T_world_vehicle": T_wv.tolist()}
        meta["frames"].append(entry)

    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "tracks.json").write_text(json.dumps(tracks, indent=1))
    return out
