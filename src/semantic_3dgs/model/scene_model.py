"""Top-level model: scene graph + optional sky, exposure and pose-refinement modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from semantic_3dgs.config import ExperimentConfig, config_from_dict, config_to_dict
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.model.exposure import ExposureCorrection
from semantic_3dgs.model.gaussians import GaussianBuffer
from semantic_3dgs.model.pose_refine import PoseRefinement
from semantic_3dgs.model.scene_graph import SceneGraph, build_node_tables
from semantic_3dgs.model.sky import SkyCubemap


class SceneModel(nn.Module):
    def __init__(
        self,
        graph: SceneGraph,
        sky: SkyCubemap | None,
        exposure: ExposureCorrection | None,
        unmodelled_track_ids: list[str],
    ):
        super().__init__()
        self.graph = graph
        self.sky = sky
        self.exposure = exposure
        self.unmodelled_track_ids = list(unmodelled_track_ids)

    @property
    def gaussians(self) -> GaussianBuffer:
        return self.graph.gaussians

    @property
    def device(self) -> torch.device:
        return self.gaussians.means.device

    @classmethod
    def build(cls, scene: DrivingScene, cfg: ExperimentConfig) -> SceneModel:
        from semantic_3dgs.init.initialize import initialize

        init = initialize(scene, cfg)
        poses, visible = build_node_tables(scene, init.node_track_ids)
        return cls._assemble(
            cfg, init.gaussians, init.node_track_ids, poses, visible,
            scene.num_images, init.unmodelled_track_ids,
        )

    @classmethod
    def _assemble(cls, cfg, gaussians, node_track_ids, poses, visible, num_images, unmodelled):
        f = cfg.features
        num_nodes, num_frames = poses.shape[:2]
        refiner = (
            PoseRefinement(num_nodes, num_frames) if f.pose_refine and num_nodes > 1 else None
        )
        graph = SceneGraph(gaussians, node_track_ids, poses, visible, refiner)
        sky = SkyCubemap(cfg.model.sky_resolution) if f.sky else None
        exposure = ExposureCorrection(num_images) if f.exposure else None
        return cls(graph, sky, exposure, unmodelled)

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str | Path, cfg: ExperimentConfig, step: int, extra: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": config_to_dict(cfg),
                "step": step,
                "node_track_ids": self.graph.node_track_ids,
                "unmodelled_track_ids": self.unmodelled_track_ids,
                "num_images": self.exposure.affine.shape[0] if self.exposure else 0,
                "state_dict": self.state_dict(),
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple[SceneModel, ExperimentConfig, dict[str, Any]]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = config_from_dict(ckpt["config"])
        state = ckpt["state_dict"]
        empty = GaussianBuffer.from_points(
            torch.zeros(0, 3), torch.ones(0), torch.zeros(0, dtype=torch.int32), cfg.model.sh_degree
        )
        empty.allocate_like(state, prefix="graph.gaussians.")
        model = cls._assemble(
            cfg, empty, ckpt["node_track_ids"], state["graph.node_poses"],
            state["graph.node_visible"], max(ckpt["num_images"], 1), ckpt["unmodelled_track_ids"],
        )
        model.load_state_dict(state)
        return model.to(device), cfg, ckpt
