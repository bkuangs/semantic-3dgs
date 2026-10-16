"""Per-object canonical point clouds. Milestone 2."""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs._stub import todo
from semantic_3dgs.data.scene import DrivingScene


def accumulate_object_points(scene: DrivingScene, track_id: str, box_margin: float) -> Tensor:
    """Gather the LiDAR points inside the track's box at every frame where it is labelled and
    express them in the box's canonical frame ``O`` (origin at the box centre, x along heading).

    Because the object is rigid, aggregating over time densifies it far beyond a single sweep.

    Returns:
        (M, 3) float32 points in the object frame.
    """
    todo("M2", "p_O = T_O_V(t) @ p_V for points selected by points_in_box")


def fill_box(size: Tensor, count: int, seed: int = 0) -> Tensor:
    """Uniform random points inside an axis-aligned box of full extents ``size`` centred at 0.

    Used to top up objects with too few LiDAR returns (distant or heavily occluded tracks).
    """
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(count, 3, generator=generator) - 0.5) * size.float()
