"""Sky / far-field model. Milestone 3.

LiDAR never returns from the sky, so without an explicit model 3DGS spawns floaters in front of
the camera to paint it. The renderer composites ``rgb = rgb_gaussians + (1 - alpha) * sky(d)``,
and a loss on the semantic sky mask pushes ``alpha -> 0`` there.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from semantic_3dgs._stub import todo

FACE_ORDER = ("+x", "-x", "+y", "-y", "+z", "-z")


class SkyCubemap(nn.Module):
    """Learnable world-frame cubemap.

    ``faces`` holds logits of shape (6, R, R, 3) in the order :data:`FACE_ORDER`; the colour is
    ``sigmoid`` of the (bilinearly) sampled logit. A face is selected by the dominant axis of the
    direction. Within a face, any consistent (u, v) parameterisation is fine.
    """

    def __init__(self, resolution: int = 64):
        super().__init__()
        self.faces = nn.Parameter(torch.zeros(6, resolution, resolution, 3))

    def forward(self, dirs: Tensor) -> Tensor:
        """Args: dirs (..., 3) world directions, not necessarily unit length.
        Returns: (..., 3) RGB in [0, 1], differentiable w.r.t. ``faces``."""
        todo("M3", "face selection by argmax |d|, project to the face, grid_sample")
