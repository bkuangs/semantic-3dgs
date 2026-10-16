"""Loss terms (Milestones 1-4) and the weighted combination used by the trainer.

All per-pixel inputs are single images: (H, W, ...) tensors. ``mask`` arguments are (H, W) bool
tensors; pixels where the mask is False must contribute neither value nor gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from semantic_3dgs._stub import todo
from semantic_3dgs.classes import IGNORE_INDEX, SKY
from semantic_3dgs.config import ExperimentConfig
from semantic_3dgs.eval.metrics import ssim_map  # noqa: F401  (provided for rgb_loss)
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.render.pipeline import RenderOutput


def rgb_loss(pred: Tensor, target: Tensor, mask: Tensor, ssim_lambda: float) -> Tensor:
    """``(1 - lambda) * L1 + lambda * (1 - SSIM)`` over masked pixels.

    Args: pred, target (H, W, 3) in [0, 1]; mask (H, W) bool.

    Masked-out pixels must receive *no* gradient - including through SSIM's window, which mixes
    neighbouring pixels (hint: what if masked pixels of ``pred`` simply equalled the target?).
    Hint: :func:`semantic_3dgs.eval.metrics.ssim_map` returns a per-pixel SSIM map.
    """
    todo("M1")


def semantic_loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy between composited logits (H, W, C) and labels (H, W) int64.

    Pixels with ``labels == IGNORE_INDEX`` or ``mask == False`` are excluded. Returns 0 (with a
    graph) if nothing is valid.
    """
    todo("M4")


def depth_loss(pred_depth: Tensor, lidar_depth: Tensor, mask: Tensor) -> Tensor:
    """L1 between rendered expected depth (H, W) and sparse LiDAR depth (H, W, 0 = no return),
    over pixels that have a return and are in ``mask``."""
    todo("M3")


def sky_loss(alpha: Tensor, sky_mask: Tensor) -> Tensor:
    """Push accumulated opacity (H, W) towards 0 on sky pixels (H, W bool). Zero when
    ``alpha == 0`` on the sky, increasing in alpha, and independent of non-sky pixels."""
    todo("M3", "e.g. binary cross-entropy against 0 on sky pixels")


def opacity_regularization(opacities: Tensor) -> Tensor:
    """Mean L1 of activated opacities (N,) - keeps the Gaussian count honest under MCMC."""
    todo("M1")


def scale_regularization(scales: Tensor) -> Tensor:
    """Mean L1 of activated scales (N, 3) - discourages huge needles / blobs."""
    todo("M1")


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

