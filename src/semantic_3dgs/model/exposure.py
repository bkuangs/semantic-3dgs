"""Per-image affine colour correction (auto-exposure / white balance). Milestone 3."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from semantic_3dgs._stub import todo


class ExposureCorrection(nn.Module):
    """``affine[i]`` is a (3, 4) matrix ``[A | b]`` applied as ``rgb' = A @ rgb + b`` to the
    rendered image of training image ``i`` before the photometric loss. Initialised to identity.

    The absolute colour scale is unobservable (a global gain can move into the Gaussians), so
    :meth:`regularization` should keep corrections near identity. Novel views are rendered
    without correction.
    """

    def __init__(self, num_images: int):
        super().__init__()
        identity = torch.eye(3, 4)
        self.affine = nn.Parameter(identity.repeat(num_images, 1, 1))

    def forward(self, rgb: Tensor, image_index: int) -> Tensor:
        """Args: rgb (H, W, 3). Returns corrected (H, W, 3)."""
        todo("M3")

    def regularization(self) -> Tensor:
        """Scalar, zero at initialisation, growing with the distance from identity."""
        todo("M3")
