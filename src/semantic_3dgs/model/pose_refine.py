"""Learnable corrections to noisy 3D box trajectories. Milestone 3."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from semantic_3dgs._stub import todo


class PoseRefinement(nn.Module):
    """Per-node, per-frame twist ``delta_xi[k, t]`` in se(3), initialised to zero.

    Convention: the refined pose is ``T_W_O(t) = T_hat_W_O(t) @ exp(delta_xi[k, t])``, i.e. the
    correction is applied in the *object* frame, so translation components read as
    "forward / left / up relative to the car". Node 0 (background) must never move.
    """

    def __init__(self, num_nodes: int, num_frames: int):
        super().__init__()
        self.deltas = nn.Parameter(torch.zeros(num_nodes, num_frames, 6))

    def forward(self, base_poses: Tensor, frame: int) -> Tensor:
        """Args: base_poses (K+1, 4, 4) at ``frame``. Returns refined (K+1, 4, 4)."""
        todo("M3", "use geometry.lie.se3_exp; keep node 0 exactly at base_poses[0]")

    def smoothness_loss(self) -> Tensor:
        """Scalar penalty on frame-to-frame changes of the corrections of dynamic nodes, so the
        refinement removes jitter instead of inventing motion."""
        todo("M3")
