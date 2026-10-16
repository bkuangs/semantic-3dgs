"""Composite scene graph: background node 0 plus one rigid node per dynamic track. Milestone 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from semantic_3dgs._stub import todo
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.geometry import box_to_pose
from semantic_3dgs.model.gaussians import GaussianBuffer
from semantic_3dgs.model.pose_refine import PoseRefinement


@dataclass
class ComposedGaussians:
    means: Tensor  # (N, 3) world frame
    quats: Tensor  # (N, 4) wxyz, world frame
    rotations: Tensor  # (N, 3, 3) R_W_node of the node each Gaussian belongs to
    visible: Tensor  # (N,) bool - False for objects that do not exist at this frame


def gather_per_gaussian(table: Tensor, node_ids: Tensor) -> Tensor:
    """``table[node_ids]`` for a small per-node table (K+1, ...) -> (N, ...).

    Plain indexing is correct, but when ``table`` requires grad (pose refinement) its backward
    scatter-adds ~10^6 rows into K+1 slots with atomics, which is ~30x slower than the whole
    rasterizer. A one-hot matmul turns the backward into a dense GEMM instead.
    """
    if not table.requires_grad:
        return table[node_ids.long()]
    onehot = torch.nn.functional.one_hot(node_ids.long(), table.shape[0]).to(table.dtype)
    return (onehot @ table.reshape(table.shape[0], -1)).reshape(len(node_ids), *table.shape[1:])


def compose_gaussians(
    means: Tensor,
    quats: Tensor,
    node_ids: Tensor,
    node_poses: Tensor,
    node_visible: Tensor,
) -> ComposedGaussians:
    """Place every Gaussian in the world frame at one timestamp.

    Args:
        means: (N, 3) node-local positions.
        quats: (N, 4) node-local wxyz orientations (not necessarily normalised).
        node_ids: (N,) int node index of each Gaussian.
        node_poses: (K+1, 4, 4) ``T_W_node`` at this timestamp (node 0 = identity background).
        node_visible: (K+1,) bool.

    Must be differentiable w.r.t. ``means``, ``quats`` and ``node_poses`` and vectorised (no
    Python loop over Gaussians; a loop over the K nodes is acceptable but unnecessary).

    Math: mu_W = R mu_O + p, q_W = q(R) ⊗ q_O  (equivalently Sigma_W = R Sigma_O R^T).

    Use :func:`gather_per_gaussian` (not ``node_poses[node_ids]``) to expand per-node quantities,
    otherwise pose refinement (M3) makes every training step ~10x slower.
    """
    todo("M2", "gather per-Gaussian transforms with node_ids, then rotate/translate")


def build_node_tables(
    scene: DrivingScene, node_track_ids: list[str | None]
) -> tuple[Tensor, Tensor]:
    """``T_W_node`` for every node and frame, plus a visibility table.

    Frames where a track is unlabelled reuse the nearest labelled pose and are marked invisible.

    Returns:
        poses: (K+1, F, 4, 4), visible: (K+1, F) bool.
    """
    num_nodes, num_frames = len(node_track_ids), scene.num_frames
    poses = torch.eye(4).repeat(num_nodes, num_frames, 1, 1)
    visible = torch.ones(num_nodes, num_frames, dtype=torch.bool)
    for node, track_id in enumerate(node_track_ids):
        if track_id is None:
            continue
        boxes = scene.tracks[track_id].boxes
        labelled = sorted(boxes)
        for frame in range(num_frames):
            nearest = min(labelled, key=lambda f, fr=frame: abs(f - fr))
            box = boxes[nearest]
            T_vehicle_box = box_to_pose(
                torch.tensor(box.center, dtype=torch.float32),
                torch.tensor(box.heading, dtype=torch.float32),
            )
            poses[node, frame] = scene.T_world_vehicle(nearest) @ T_vehicle_box
            visible[node, frame] = frame in boxes
    return poses, visible


class SceneGraph(nn.Module):
    def __init__(
        self,
        gaussians: GaussianBuffer,
        node_track_ids: list[str | None],
        node_poses: Tensor,
        node_visible: Tensor,
        pose_refiner: PoseRefinement | None = None,
    ):
        super().__init__()
        self.gaussians = gaussians
        self.node_track_ids = list(node_track_ids)
        self.register_buffer("node_poses", node_poses)
        self.register_buffer("node_visible", node_visible)
        self.pose_refiner = pose_refiner

    @property
    def num_nodes(self) -> int:
        return len(self.node_track_ids)

    def node_index(self, track_id: str) -> int:
        return self.node_track_ids.index(track_id)

    def poses_at(self, frame: int) -> Tensor:
        poses = self.node_poses[:, frame]
        if self.pose_refiner is not None:
            poses = self.pose_refiner(poses, frame)
        return poses

    def forward(self, frame: int, node_poses: Tensor | None = None) -> ComposedGaussians:
        """Compose the scene at ``frame``. ``node_poses`` overrides the (refined) track poses,
        which is how counterfactual trajectories are rendered."""
        poses = self.poses_at(frame) if node_poses is None else node_poses
        g = self.gaussians
        return compose_gaussians(
            g.means, g.quats, g.node_ids.long(), poses, self.node_visible[:, frame]
        )
