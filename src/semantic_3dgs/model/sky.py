"""Sky / far-field model. Milestone 3.

LiDAR never returns from the sky, so without an explicit model 3DGS spawns floaters in front of
the camera to paint it. The renderer composites ``rgb = rgb_gaussians + (1 - alpha) * sky(d)``,
and a loss on the semantic sky mask pushes ``alpha -> 0`` there.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

import torch.nn.functional as F

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
        """Take a viewing direction and return the sky's color in that direction. Sky is a learnable
        cube of 6 small images surrounding the scene; each direction looks up the pixel where its ray 
        would hit the cube.

        We just pick the direction with the highest magnitude.

        For example, (1.0, 0.1, −0.2) has its largest component along x, and it's positive, so it
        hits face 0 (+x). (0, 0, 1) hits face 4 (+z, straight up).
        
        Args: dirs (..., 3) world directions, not necessarily unit length.
        Returns: (..., 3) RGB in [0, 1], differentiable w.r.t. ``faces``.
        """
        shape = dirs.shape[:-1]
        d = dirs.reshape(-1, 3)

        # Pick the face depending on which axis the direction points along the most
        axis = d.abs().argmax(dim=-1)
        major = d.gather(-1, axis[: None]).squeeze(-1)
        face = 2 * axis + (major < 0).long()

        others = torch.tensor([[1, 2], [0, 2], [0, 1]])[axis]
        uv = d.gather(-1, others) / major.abs().clamp_min(1e-8)[:, None]    # perspective divide

        # Loop through all faces because different pixels in one frame have different directions
        for f in range(6):
            m = face == f
            if not m.any():
                continue

            img = self.faces[f].permute(2, 0, 1)[None]  # (6, R, R, 3)->(R, R, 3)->(3, R, R)->(1, 3, R, R)
            grid = uv[m][None, None]
            s = F.grid_sample(img, frid, mode="bilinear", padding_mode="border", align_corners=False)
            out[m] = s[0, :, 0].T                       # remove extra dim and put color last again

        return torch.sigmoid(out).reshape(*shape, 3)
