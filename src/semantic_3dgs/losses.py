"""Loss terms (Milestones 1-4) and the weighted combination used by the trainer.

All per-pixel inputs are single images: (H, W, ...) tensors. ``mask`` arguments are (H, W) bool
tensors; pixels where the mask is False must contribute neither value nor gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from semantic_3dgs._stub import todo
from semantic_3dgs.classes import IGNORE_INDEX, SKY
from semantic_3dgs.config import ExperimentConfig
from semantic_3dgs.eval.metrics import ssim_map  # noqa: F401  (provided for rgb_loss)
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.render.pipeline import RenderOutput


def rgb_loss(pred: Tensor, target: Tensor, mask: Tensor, ssim_lambda: float) -> Tensor:
    """Photometric loss: how far the rendered image is from the real photo. Two terms:
        * L1: the per-pixel color error.
        * 1 - SSIM: a structural similarity term that compares local patches (contrast, edges, texture).
            * SSIM = Structural Similarity Index Measure

    The overall loss function combines them.

    Args: pred, target (H, W, 3) in [0, 1]; mask (H, W) bool.

    Computed only over pixels where mask is True; masked-out pixels get no gradient.

    Masking rule: a track is masked if it moves but doesn't get its own node, eg. pedestrians
                  that we can't model.
    """
    if not mask.any():
        return pred.sum() * 0.0  # zero, but still part of the graph

    # SSIM looks at a window, so the gradient would leak through 
    # Replace masked-out pixels with the target, so no gradient can flow back and it doesn't pull loss
    pred = torch.where(mask[..., None], pred, target)

    l1 = (pred - target).abs()[mask].mean()              # mean over valid pixels and channels
    if ssim_lambda == 0:
        return l1

    ssim = ssim_map(pred, target)[mask].mean()           # ssim_map -> (H, W)

    return (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim)


def semantic_loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy between composited logits (H, W, C) and labels (H, W) int64.

    Pixels with ``labels == IGNORE_INDEX`` or ``mask == False`` are excluded. Returns 0 (with a
    graph) if nothing is valid.
    """
    todo("M4")


def depth_loss(pred_depth: Tensor, lidar_depth: Tensor, mask: Tensor) -> Tensor:
    """RGB loss only checks colors. Multiple 3D arrangements can produce the same image,
    especially on textureless surfaces. So a model can get colors right but geometry wrong.

    Solution: Use LiDAR to measure true distance for some pixels. The loss is the L1 error
    between rendered depth and LiDAR depth.
    """
    valid = mask & (lidar_depth > 0)
    if not valid.any():                 # no lidar hits shouldn't return NaN
        return pred_depth.sum() * 0.0

    return (pred_depth - lidar_depth).abs()[valid].mean()


def sky_loss(alpha: Tensor, sky_mask: Tensor) -> Tensor:
    """Without guidance, the optimizer paints the sky with Gaussians giving garbage depth.

    Solution: On pixels segmented as sky, pushed rendered opacity to 0, which shows the sky
    cubemap instead.
    """
    if not sky_mask.any():
        return alpha.sum() * 0.0        # no sky pixels shouldn't return NaN

    a = alpha[sky_mask].clamp(max=1 - 1e-6)

    return -torch.log1p(-a).mean()


def opacity_regularization(opacities: Tensor) -> Tensor:
    """Mean L1 of activated opacities (N,) - keeps the Gaussian count honest under MCMC."""
    return opacities.abs().mean()


def scale_regularization(scales: Tensor) -> Tensor:
    """Mean L1 of activated scales (N, 3) - discourages huge needles / blobs."""
    return scales.abs().mean()


@dataclass
class Batch:
    image: Tensor  # (H, W, 3)
    mask: Tensor  # (H, W) bool: valid after undistortion and not an unmodelled actor
    semantics: Tensor | None  # (H, W) int64
    lidar_depth: Tensor | None  # (H, W)


def compute_losses(
    out: RenderOutput, batch: Batch, model: SceneModel, cfg: ExperimentConfig
) -> dict[str, Tensor]:
    t, f = cfg.train, cfg.features
    terms: dict[str, Tensor] = {"rgb": rgb_loss(out.rgb, batch.image, batch.mask, t.ssim_lambda)}
    total = terms["rgb"]

    labels = batch.semantics
    if labels is not None and f.semantics and out.semantics is not None:
        sem_mask = batch.mask & (labels != SKY) & (labels != IGNORE_INDEX)
        terms["semantic"] = semantic_loss(out.semantics, labels, sem_mask)
        total = total + t.semantic_weight * terms["semantic"]
    if labels is not None and f.sky:
        terms["sky"] = sky_loss(out.alpha, (labels == SKY) & batch.mask)
        total = total + t.sky_weight * terms["sky"]
    if batch.lidar_depth is not None and f.depth_loss:
        terms["depth"] = depth_loss(out.depth, batch.lidar_depth, batch.mask)
        total = total + t.depth_weight * terms["depth"]

    g = model.gaussians
    if t.strategy == "mcmc":
        terms["opacity_reg"] = opacity_regularization(g.opacities)
        terms["scale_reg"] = scale_regularization(g.scales)
        total = total + t.opacity_reg_weight * terms["opacity_reg"]
        total = total + t.scale_reg_weight * terms["scale_reg"]
    if model.exposure is not None:
        terms["exposure_reg"] = model.exposure.regularization()
        total = total + t.exposure_reg_weight * terms["exposure_reg"]
    if model.graph.pose_refiner is not None:
        terms["pose_smooth"] = model.graph.pose_refiner.smoothness_loss()
        total = total + t.pose_smooth_weight * terms["pose_smooth"]

    terms["total"] = total
    return terms

