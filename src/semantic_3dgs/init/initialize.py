"""Turn point clouds into an initial :class:`GaussianBuffer` and scene-graph node list."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from semantic_3dgs.config import ExperimentConfig
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.init.lidar import (
    accumulate_static_points,
    classify_dynamic_tracks,
    random_subsample,
    voxel_downsample,
)
from semantic_3dgs.init.objects import accumulate_object_points, fill_box
from semantic_3dgs.model.gaussians import GaussianBuffer

log = logging.getLogger(__name__)


@dataclass
class Initialization:
    gaussians: GaussianBuffer
    node_track_ids: list[str | None]  # node 0 is the background (None)
    unmodelled_track_ids: list[str]  # tracked actors whose pixels are excluded from the loss


def knn_scales(points: Tensor, k: int = 3, max_scale: float | None = None) -> Tensor:
    """Isotropic initial scale = RMS distance to the ``k`` nearest neighbours."""
    if len(points) <= k:
        return torch.full((len(points),), 0.05)
    tree = cKDTree(points.numpy())
    dist, _ = tree.query(points.numpy(), k=k + 1)
    scale = torch.from_numpy(np.sqrt((dist[:, 1:] ** 2).mean(1))).float().clamp_min(1e-4)
    if max_scale is not None:
        scale = scale.clamp_max(max_scale)
    return scale


def initialize(scene: DrivingScene, cfg: ExperimentConfig) -> Initialization:
    icfg = cfg.init
    features = cfg.features

    candidates = classify_dynamic_tracks(
        scene, icfg.dynamic_classes, icfg.dynamic_displacement_threshold
    )
    dynamic_ids = sorted(candidates) if features.dynamic else []
    # Tracks with moving boxes that are not modelled as nodes (e.g. pedestrians, or every moving
    # vehicle while the dynamic branch is disabled) are removed from the background cloud.
    moving_any = classify_dynamic_tracks(
        scene, tuple(sorted({t.category for t in scene.tracks.values()})),
        icfg.dynamic_displacement_threshold,
    )
    unmodelled = sorted(set(moving_any) - set(dynamic_ids))
    excluded = sorted(set(moving_any) | set(dynamic_ids))

    bg = accumulate_static_points(scene, excluded, icfg.box_margin)
    bg = voxel_downsample(bg, icfg.voxel_size)
    bg = random_subsample(bg, icfg.max_background_points, seed=cfg.train.seed)
    log.info("background: %d points after voxel downsampling", len(bg))

    clouds = [bg]
    node_ids = [torch.zeros(len(bg), dtype=torch.int32)]
    for node, track_id in enumerate(dynamic_ids, start=1):
        pts = accumulate_object_points(scene, track_id, icfg.box_margin)
        pts = voxel_downsample(pts, icfg.voxel_size / 2)
        if len(pts) < icfg.min_object_points:
            size = torch.from_numpy(next(iter(scene.tracks[track_id].boxes.values())).size)
            extra = fill_box(size, icfg.min_object_points - len(pts), seed=node)
            pts = torch.cat([pts, extra])
        clouds.append(pts)
        node_ids.append(torch.full((len(pts),), node, dtype=torch.int32))
        log.info("node %d (%s): %d points", node, track_id, len(pts))

    scales = torch.cat([knn_scales(c, max_scale=1.0) for c in clouds])
    gaussians = GaussianBuffer.from_points(
        means=torch.cat(clouds),
        scales=scales,
        node_ids=torch.cat(node_ids),
        sh_degree=cfg.model.sh_degree,
        initial_opacity=icfg.initial_opacity,
        num_semantic_classes=_num_classes(cfg),
    )
    return Initialization(gaussians, [None, *dynamic_ids], unmodelled)


def _num_classes(cfg: ExperimentConfig) -> int:
    from semantic_3dgs.classes import NUM_CLASSES

    return NUM_CLASSES if cfg.features.semantics else 0
