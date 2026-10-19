"""Learnable corrections to noisy 3D box trajectories. Milestone 3."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from semantic_3dgs.geometry.lie import se3_exp


class PoseRefinement(nn.Module):
    """Learned corrections that cancel the jitter in 3D box labels.

    Learn a small correction per car per frame (photometric loss) that moves 
    each car to where the images show it.

    How a correction is applied (``T_hat`` = labelled pose, ``T`` = refined pose):

        T_W_O(t) = T_hat_W_O(t) @ se3_exp(deltas[k, t])

    The background (Node 0) is never corrected, otherwise the whole world could drift.
     
    smoothness_loss() discourages corrections that jump between neighbouring frames,
    so the module cancels small jitter instead of inventing motion.
    """

    def __init__(self, num_nodes: int, num_frames: int):
        super().__init__()
        self.deltas = nn.Parameter(torch.zeros(num_nodes, num_frames, 6))   # learnable pose corrections

    def forward(self, base_poses: Tensor, frame: int) -> Tensor:
        """Take the labelled poses for one frame and return slightly corrected poses.
        One learned correction per node applied in each node's own frame.
        
        Args: 
        * base_poses: the queried poses from the node table
        * frame: the frame to correct
        
        Returns refined (K+1, 4, 4).
        """
        xi = self.deltas[:, frame]      # one twist (ρ, φ) per node
        D = se3_exp(xi)                 # 6 num -> rigid tf
        refined = base_poses @ D        # apply correction

        return torch.cat([base_poses[:1], refined[1:]], dim=0)

    def smoothness_loss(self) -> Tensor:
        """
        Penalize sharp corrections so they stay small and steady.
        """
        d = self.deltas[1:]
        diff = d[:, 1:] - d[:, :-1]
        if diff.numel() == 0:
            return self.deltas.sum() * 0.0

        return diff.pow(2).mean()
