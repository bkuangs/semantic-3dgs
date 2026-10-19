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
from semantic_3dgs.geometry.lie import quat_multiply, rotmat_to_quat


@dataclass
class ComposedGaussians:
    means: Tensor  # (N, 3) world frame
    quats: Tensor  # (N, 4) wxyz, world frame
    rotations: Tensor  # (N, 3, 3) R_W_node of the node each Gaussian belongs to
    visible: Tensor  # (N,) bool - False for objects that do not exist at this frame


def gather_per_gaussian(table: Tensor, node_ids: Tensor) -> Tensor:
    """Copy each node's data to every Gaussian belonging to that node, so that any transform
    applied to the node is applied to the Gaussians as well (in a batched operation).

    Example: nodes = [background, car_A], node_ids = [0, 0, 1, 1, 1]:
    * Gaussians 0 and 1 get the identity pose, so they stay where they are, in world coordinates.
    * Gaussians 2, 3, and 4 get car_A's pose for this frame, so they move with the car.

    The copy is cheap, and it's rebuilt every frame because the car poses change from frame to frame.
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
    """Assemble the entire scene at one frame: Move every Gaussian from its node local frame
    into the world frame, so the rasterizer can draw them all together.

    The rasterizer (gsplat) only accepts one flat list of Gaussians in world coordinates, so we
    must transform each car's Gaussians into the world frame.

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
    # Get the Gaussians of each node
    visible = gather_per_gaussian(node_visible, node_ids)
    poses = gather_per_gaussian(node_poses, node_ids)
    
    R, p = poses[:, :3, :3], poses[:, :3, 3]                # split pose

    means = (R @ means.unsqueeze(-1)).squeeze(-1) + p       # move to world frame

    node_q = rotmat_to_quat(node_poses[:, :3, :3])
    q_g = gather_per_gaussian(node_q, node_ids)
    quats = quat_multiply(q_g, quats)                       # rotate Gaussian with its node

    return ComposedGaussians(means, quats, R, visible)


def build_node_tables(
    scene: DrivingScene, node_track_ids: list[str | None]
) -> tuple[Tensor, Tensor]:
    """Precompute where every node is in the world at every frame, plus whether it 
    exists in that frame. Runs once, before training, and produces two lookup tables:
    * poses:   Where node k is in the world at frame t
    * visible: Whether node k is labelled at frame t

    For example, a background and 2 cars over 4 frames:

                    frame 0     frame 1     frame 2     frame 3
        node 0 bg   I  ✓        I  ✓        I  ✓        I  ✓      ← identity, always visible
        node 1 car  T₀ ✓        T₁ ✓        T₂ ✓        T₂ ✗      ← left after frame 2
        node 2 car  T₁' ✗       T₁' ✓       T₂' ✓       T₃' ✓     ← entered at frame 1

    Steps:
    1. Look up the car's box at frame t
    2. Transform box -> vehicle -> world
    3. Mark visible if the car has a box at frame t
        a. This is a dataset lookup (dataset labelled this object at this sweep)

    Frames where a track is unlabelled reuse the nearest labelled pose and are marked invisible.
    This allows every entry holds a valid pose, so the tensor has no holes.

    Consumption: Tells the renderer where to draw each object and whether to draw it at each frame.

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
    """
    Node: One piece of the scene that moves as a unit, has its own coordinate frame and pose.
        * Node 0:        Static background (road, buildings, trees, etc.)
        * Nodes 1 ... K: Moving objects, eg. driving car

    Every Gaussian belongs to exactly 1 node, node_ids[i] index.

    Appearance (Gaussians) are stored separately from motion (pose), so we learn the car
    once and then move it anywhere.
    """
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
