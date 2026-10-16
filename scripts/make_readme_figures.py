"""Regenerate the README figures in ``docs/images`` from trained synthetic runs.

Needs a working implementation of every milestone (the scaffold's stubs raise), so run it once
you have finished M5::

    uv run s3dgs synth
    uv run s3dgs train configs/synthetic/m1_static.yaml
    uv run s3dgs train configs/synthetic/m4_semantic.yaml
    uv run python scripts/make_readme_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from semantic_3dgs import viz
from semantic_3dgs.classes import CLASSES, PALETTE
from semantic_3dgs.config import DataConfig
from semantic_3dgs.counterfactual import heading_from_lateral, offset_pose, smoothstep_profile
from semantic_3dgs.data.depth import lidar_depth_image
from semantic_3dgs.data.masks import actor_mask
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.data.splits import split_images
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.render.pipeline import View, render, view_from_image

CAMERAS = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(image: np.ndarray, text: str, size: int = 16) -> np.ndarray:
    """Add a dark caption bar above an image."""
    bar = Image.new("RGB", (image.shape[1], size + 10), (24, 24, 24))
    ImageDraw.Draw(bar).text((6, 4), text, fill=(235, 235, 235), font=_font(size))
    return np.concatenate([np.asarray(bar), image], 0)


def grid(rows: list[list[np.ndarray]], pad: int = 4) -> np.ndarray:
    def hcat(images):
        sep = np.full((images[0].shape[0], pad, 3), 255, np.uint8)
        out = [images[0]]
        for im in images[1:]:
            out += [sep, im]
        return np.concatenate(out, 1)

    lines = [hcat(r) for r in rows]
    sep = np.full((pad, lines[0].shape[1], 3), 255, np.uint8)
    out = [lines[0]]
    for line in lines[1:]:
        out += [sep, line]
    return np.concatenate(out, 0)


def legend(width: int) -> np.ndarray:
    bar = Image.new("RGB", (width, 26), (255, 255, 255))
    draw, font, x = ImageDraw.Draw(bar), _font(12), 6
    for name, color in zip(CLASSES, PALETTE, strict=True):
        draw.rectangle([x, 7, x + 12, 19], fill=tuple(int(c) for c in color), outline=(90, 90, 90))
        draw.text((x + 16, 6), name, fill=(30, 30, 30), font=font)
        x += 22 + int(draw.textlength(name, font=font))
    return np.asarray(bar)


def _depth(out) -> np.ndarray:
    return viz.colorize_depth(out.depth, max_depth=60.0, valid=out.alpha > 0.5)


def inputs_figure(scene: DrivingScene, frame: int) -> np.ndarray:
    index = {scene.images[i].camera: i for i in range(scene.num_images) if scene.images[i].frame == frame}
    rgb, sem, check = [], [], []
    for cam in CAMERAS:
        i = index[cam]
        image = scene.image(i)
        rgb.append(label(viz.to_uint8(image), f"{cam} camera"))
        sem.append(label(viz.colorize_labels(scene.semantics(i)), "2D semantic labels"))
        overlay = viz.overlay_points(image, lidar_depth_image(scene, i))
        overlay = viz.overlay_mask(overlay, actor_mask(scene, i, list(scene.tracks), 0.0))
        check.append(label(overlay, "LiDAR + tracked boxes (M0)"))
    out = grid([rgb, sem, check])
    return np.concatenate([out, legend(out.shape[1])], 0)


def comparison_figure(m1: SceneModel, m4: SceneModel, scene: DrivingScene, index: int, cfg) -> np.ndarray:
    gt = viz.to_uint8(scene.image(index))
    with torch.no_grad():
        o1 = render(m1, view_from_image(scene, index), cfg.render)
        o4 = render(m4, view_from_image(scene, index), cfg.render)
    top = [label(gt, "Ground truth (held-out view)"), label(viz.to_uint8(o1.rgb), "M1: static background only"),
           label(viz.to_uint8(o4.rgb), "M4: scene graph + robustness")]
    bottom = [label(_depth(o4), "M4 rendered depth"),
              label(viz.colorize_labels(viz.semantic_prediction(o4.semantics, o4.alpha)), "M4 rendered semantics"),
              label(viz.colorize_labels(scene.semantics(index)), "2D labels")]
    out = grid([top, bottom])
    return np.concatenate([out, legend(out.shape[1])], 0)


def counterfactual_frames(model: SceneModel, scene: DrivingScene, cfg, track: str, ego_left: float,
                          actor_left: float, start: int, duration: int, scale: int = 2) -> list[np.ndarray]:
    cam = scene.cameras["FRONT"]
    node = model.graph.node_index(track)
    profile = smoothstep_profile(scene.num_frames, start, duration, actor_left)
    speed = torch.linalg.norm(torch.gradient(model.graph.node_poses[node, :, :3, 3].cpu(), dim=0)[0], dim=-1)
    yaws = heading_from_lateral(profile, speed)
    frames = []
    for frame in range(scene.num_frames):
        rows = []
        for edited in (False, True):
            poses = model.graph.poses_at(frame).clone()
            T_wv = scene.T_world_vehicle(frame)
            if edited:
                poses[node] = offset_pose(poses[node], left=float(profile[frame]), yaw=float(yaws[frame]))
                T_wv = offset_pose(T_wv, left=ego_left)
            view = View(K=scene.K("FRONT"), width=cam.width, height=cam.height, T_world_vehicle=T_wv,
                        T_vehicle_camera=scene.T_vehicle_camera("FRONT"), frame=frame)
            with torch.no_grad():
                out = render(model, view, cfg.render, node_poses=poses, apply_exposure=False)
            panels = [viz.to_uint8(out.rgb), viz.colorize_labels(viz.semantic_prediction(out.semantics, out.alpha)),
                      _depth(out)]
            panels = [p[::scale, ::scale] for p in panels]
            title = (f"Counterfactual: ego {ego_left:+.0f} m left, {track} cut-in" if edited
                     else "Logged trajectory (reconstruction)")
            rows.append(label(viz.hstack(*panels), title, size=13))
        frames.append(np.concatenate([rows[0], np.full((4, rows[0].shape[1], 3), 255, np.uint8), rows[1]], 0))
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m1", default="outputs/synthetic/m1_static")
    p.add_argument("--m4", default="outputs/synthetic/m4_semantic")
    p.add_argument("--out", default="docs/images")
    p.add_argument("--frame", type=int, default=12)
    p.add_argument("--track", default="cut_in_car")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def load(run: str) -> tuple[SceneModel, object]:
        model, cfg, _ = SceneModel.load(sorted(Path(run, "checkpoints").glob("step_*.pt"))[-1], "cuda")
        return model, cfg

    m1, _ = load(args.m1)
    m4, cfg = load(args.m4)
    scene = DrivingScene(cfg.data)
    raw = DrivingScene(DataConfig(scene_dir=cfg.data.scene_dir, downscale=cfg.data.downscale, cache_images=False))

    viz.save_png(inputs_figure(raw, args.frame), out / "inputs.png")

    _, test = split_images(scene, cfg.data.holdout_every)
    front = [i for i in test if scene.images[i].camera == "FRONT"]
    index = min(front, key=lambda i: abs(scene.images[i].frame - args.frame))
    viz.save_png(comparison_figure(m1, m4, scene, index, cfg), out / "results.png")

    frames = counterfactual_frames(m4, scene, cfg, args.track, ego_left=2.0, actor_left=1.5, start=5, duration=15)
    imageio.mimsave(out / "counterfactual.gif", frames, duration=100, loop=0)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
