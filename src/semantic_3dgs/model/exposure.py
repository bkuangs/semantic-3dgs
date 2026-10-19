"""Per-image affine colour correction (auto-exposure / white balance). Milestone 3."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from semantic_3dgs._stub import todo


class ExposureCorrection(nn.Module):
    """Problem: Each camera runs auto-exposure and auto-white-balance, so the same
    object can come out brighter/warmer/darker between images/cameras. Gaussians have 
    one color per viewing direction, so if two photos disagree, the model comprimises.

    Solution: Give each training image its own small color transform, applied to the
    rendered image before it's compared to the photo.

    Example: A = diag(2, 2, 2) and b = (0.1, 0.1, 0.1)

        pixel (0.2, 0.3, 0.4) → (2·0.2 + 0.1, 2·0.3 + 0.1, 2·0.4 + 0.1) = (0.5, 0.7, 0.9)
    """

    def __init__(self, num_images: int):
        super().__init__()
        identity = torch.eye(3, 4)
        self.affine = nn.Parameter(identity.repeat(num_images, 1, 1))   # [A | b], starting at A=I, b=0

    def forward(self, rgb: Tensor, image_index: int) -> Tensor:
        """Args: rgb (H, W, 3). Returns corrected (H, W, 3)."""
        M = self.affine[image_index]
        A = M[:, :3]
        b = M[:, 3]

        return rgb @ A.T + b

    def regularization(self) -> Tensor:
        """Keep each correction close to the identity, so it only explains real differences 
        between images.
        """
        diff = self.affine - identity

        return diff.pow(2).mean()
