"""Per-object canonical point clouds. Milestone 2."""

from __future__ import annotations

import torch
from torch import Tensor

from semantic_3dgs._stub import todo
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.geometry import box_to_pose, points_in_box, transform_points


def accumulate_object_points(scene: DrivingScene, track_id: str, box_margin: float) -> Tensor:
    """Build the starting point cloud for one moving object in its own frame. This will seed the 
    car's Gaussians in the scene graph.

    A single LiDAR sweep only catches a few hundred points on a car, mostly on the side facing the sensor. 
    But the car is rigid, so its shape doesn't change between frames. If you collect its points from every 
    frame and express them all relative to the car itself, they stack up into one much denser model of the 
    car, with different sides filled in as it moves past.

    Because the object is rigid, aggregating over time densifies it far beyond a single sweep.

    Returns:
        (M, 3) float32 points in the object frame.
    """
    chunks = []

    for frame, box in scene.tracks[track_id].boxes.items():   # only frames where it's labelled
        pts = scene.lidar(frame)

        T_vo = box_to_pose(torch.as_tensor(box.center, dtype=torch.float32),    # box -> ego frame
                            torch.tensor(box.heading, dtype=torch.float32))
        size = torch.as_tensor(box.size, dtype=torch.float32)
        inside = points_in_box(pts, T_vo, size, margin=box_margin)
        
    chunks.append(transform_points(invert_se3(T_vo), pts[inside]))
    if not chunks: return torch.zeros(0, 3)

    return torch.cat(chunks).float()


def fill_box(size: Tensor, count: int, seed: int = 0) -> Tensor:
    """Uniform random points inside an axis-aligned box of full extents ``size`` centred at 0.

    Used to top up objects with too few LiDAR returns (distant or heavily occluded tracks).
    """
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(count, 3, generator=generator) - 0.5) * size.float()
