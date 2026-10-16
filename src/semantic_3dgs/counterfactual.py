"""Counterfactual rendering: shift the ego camera and edit actor trajectories. Milestone 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from semantic_3dgs import viz
from semantic_3dgs._stub import todo
from semantic_3dgs.config import ExperimentConfig
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.render.pipeline import View, render


def offset_pose(
    T_world_x: Tensor, forward: float | Tensor = 0.0, left: float | Tensor = 0.0,
    up: float | Tensor = 0.0, yaw: float | Tensor = 0.0,
) -> Tensor:
    """Move a frame by an offset expressed in *its own* axes (x forward, y left, z up) and rotate
    it by ``yaw`` about its own z axis. Used both for the ego lane shift and for actor edits.

    Args:
        T_world_x: (4, 4) pose.
    Returns:
        (4, 4) new pose.
    """
    todo("M5", "right-multiply by a local transform")


def smoothstep_profile(num_frames: int, start: int, duration: int, magnitude: float) -> Tensor:
    """0 before ``start``, ``magnitude`` after ``start + duration``, C1-smooth in between."""
    t = torch.arange(num_frames, dtype=torch.float32)
    s = ((t - start) / max(duration, 1)).clamp(0, 1)
    return magnitude * s * s * (3 - 2 * s)


def heading_from_lateral(profile: Tensor, speed_m_per_frame: Tensor) -> Tensor:
    """Yaw that keeps an edited actor pointing along its new path (small-angle approximation)."""
    dlat = torch.gradient(profile)[0]
    return torch.atan2(dlat, speed_m_per_frame.clamp_min(1e-3))


@torch.no_grad()
def render_counterfactual(
    model: SceneModel,
    scene: DrivingScene,
    cfg: ExperimentConfig,
    out_path: Path,
    camera: str = "FRONT",
    ego_left: float = 0.0,
    track_id: str | None = None,
    actor_left: float = 0.0,
    actor_start: int = 0,
    actor_duration: int = 20,
    fps: int = 10,
) -> Path:
    """Render RGB | semantics | depth panels along a modified trajectory and write an mp4."""
    cam = scene.cameras[camera]
    profile = smoothstep_profile(scene.num_frames, actor_start, actor_duration, actor_left)
    node = model.graph.node_index(track_id) if track_id is not None else None
    if node is not None:
        positions = model.graph.node_poses[node, :, :3, 3].cpu()
        speed = torch.linalg.norm(torch.gradient(positions, dim=0)[0], dim=-1)
        yaws = heading_from_lateral(profile, speed)

    frames: list[np.ndarray] = []
    for frame in range(scene.num_frames):
        node_poses = model.graph.poses_at(frame).clone()
        if node is not None:
            node_poses[node] = offset_pose(
                node_poses[node], left=float(profile[frame]), yaw=float(yaws[frame])
            )
        view = View(
            K=scene.K(camera), width=cam.width, height=cam.height,
            T_world_vehicle=offset_pose(scene.T_world_vehicle(frame), left=ego_left),
            T_vehicle_camera=scene.T_vehicle_camera(camera), frame=frame,
        )
        out = render(model, view, cfg.render, node_poses=node_poses, apply_exposure=False)
        panels = [viz.to_uint8(out.rgb)]
        if out.semantics is not None:
            panels.append(viz.colorize_labels(viz.semantic_prediction(out.semantics, out.alpha)))
        panels.append(viz.colorize_depth(out.depth, valid=out.alpha > 0.5))
        frames.append(viz.hstack(*panels))
    viz.write_video(frames, out_path, fps=fps)
    return out_path

