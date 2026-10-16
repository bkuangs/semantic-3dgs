"""Full forward pass: compose scene graph -> SH -> gsplat rasterisation -> sky -> exposure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from semantic_3dgs.config import RenderConfig
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.geometry import invert_se3, pixel_ray_directions, world_to_camera_viewmat
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.model.sh import evaluate_sh, object_frame_view_dirs


@dataclass
class View:
    K: Tensor  # (3, 3)
    width: int
    height: int
    T_world_vehicle: Tensor  # (4, 4) ego pose
    T_vehicle_camera: Tensor  # (4, 4) Waymo-axes extrinsic
    frame: int  # which frame's object poses to use
    image_index: int | None = None  # training image index (for exposure); None = novel view

    def to(self, device) -> View:
        return View(
            self.K.to(device), self.width, self.height, self.T_world_vehicle.to(device),
            self.T_vehicle_camera.to(device), self.frame, self.image_index,
        )


def view_from_image(scene: DrivingScene, index: int) -> View:
    rec = scene.images[index]
    cam = scene.cameras[rec.camera]
    return View(
        K=scene.K(rec.camera),
        width=cam.width,
        height=cam.height,
        T_world_vehicle=scene.image_T_world_vehicle(index),
        T_vehicle_camera=scene.T_vehicle_camera(rec.camera),
        frame=rec.frame,
        image_index=index,
    )


@dataclass
class RenderOutput:
    rgb: Tensor  # (H, W, 3) final image
    alpha: Tensor  # (H, W) accumulated Gaussian opacity
    depth: Tensor  # (H, W) expected depth (meaningless where alpha ~ 0)
    semantics: Tensor | None  # (H, W, C) composited logits
    info: dict[str, Any] = field(default_factory=dict)


def render(
    model: SceneModel,
    view: View,
    cfg: RenderConfig,
    sh_degree: int | None = None,
    node_poses: Tensor | None = None,
    apply_exposure: bool = True,
) -> RenderOutput:
    from gsplat import rasterization

    device = model.device
    view = view.to(device)
    g = model.gaussians
    sh_degree = g.sh_degree if sh_degree is None else sh_degree

    viewmat = world_to_camera_viewmat(view.T_world_vehicle, view.T_vehicle_camera)
    T_world_cv = invert_se3(viewmat)
    composed = model.graph(view.frame, node_poses)

    dirs = object_frame_view_dirs(composed.means, T_world_cv[:3, 3], composed.rotations)
    features = [evaluate_sh(g.sh_coeffs, dirs, sh_degree)]
    num_sem = 0
    if g.semantics is not None:
        features.append(g.semantics)
        num_sem = g.semantics.shape[1]
    features = torch.cat(features, dim=-1)

    colors, alphas, info = rasterization(
        means=composed.means,
        quats=composed.quats,
        scales=g.scales,
        opacities=g.opacities * composed.visible,
        colors=features,
        viewmats=viewmat[None],
        Ks=view.K[None],
        width=view.width,
        height=view.height,
        near_plane=cfg.near_plane,
        far_plane=cfg.far_plane,
        render_mode="RGB+ED",
        packed=cfg.packed,
        rasterize_mode=cfg.rasterize_mode,
        sh_degree=None,
    )
    colors, alpha = colors[0], alphas[0, ..., 0]
    rgb = colors[..., :3]
    semantics = colors[..., 3 : 3 + num_sem] if num_sem else None
    depth = colors[..., -1]

    if model.sky is not None:
        rays = pixel_ray_directions(view.K, T_world_cv, view.height, view.width)
        background = model.sky(rays)
    else:
        background = torch.tensor(cfg.background, device=device)
    rgb = rgb + (1.0 - alpha)[..., None] * background

    if apply_exposure and model.exposure is not None and view.image_index is not None:
        rgb = model.exposure(rgb, view.image_index)

    return RenderOutput(rgb=rgb, alpha=alpha, depth=depth, semantics=semantics, info=info)
