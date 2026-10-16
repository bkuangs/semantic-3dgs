"""Evaluate a model on held-out images: PSNR/SSIM/LPIPS, dynamic-region PSNR, mIoU, depth."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from semantic_3dgs import viz
from semantic_3dgs.classes import IGNORE_INDEX, NUM_CLASSES
from semantic_3dgs.config import ExperimentConfig
from semantic_3dgs.data.depth import lidar_depth_image
from semantic_3dgs.data.masks import actor_mask
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.data.splits import split_images
from semantic_3dgs.eval import metrics as M
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.render.pipeline import render, view_from_image


def nearest_training_image(scene: DrivingScene, index: int, train: list[int]) -> int | None:
    rec = scene.images[index]
    same = [i for i in train if scene.images[i].camera == rec.camera]
    return min(same, key=lambda i: abs(scene.images[i].frame - rec.frame)) if same else None


@torch.no_grad()
def evaluate(
    model: SceneModel,
    scene: DrivingScene,
    indices: list[int],
    cfg: ExperimentConfig,
    out_dir: Path | None = None,
    use_lpips: bool = False,
) -> dict[str, float]:
    device = model.device
    lpips = M.LPIPS(device) if use_lpips else None
    dynamic_ids = [t for t in model.graph.node_track_ids if t is not None]
    rows: dict[str, list[float]] = {}
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    def add(key: str, value: float) -> None:
        if not math.isnan(value):
            rows.setdefault(key, []).append(value)

    train, _ = split_images(scene, cfg.data.holdout_every)
    for index in indices:
        rec = scene.images[index]
        view = view_from_image(scene, index)
        # Held-out images have no trained exposure; borrow the nearest training image's (same camera).
        view.image_index = nearest_training_image(scene, index, train)
        out = render(model, view, cfg.render)
        gt = scene.image(index).to(device)
        valid = scene.valid_mask(rec.camera).to(device)
        unmodelled = actor_mask(scene, index, model.unmodelled_track_ids).to(device)
        eval_mask = valid & ~unmodelled

        add("psnr", M.psnr(out.rgb, gt, eval_mask))
        add("ssim", M.ssim(out.rgb, gt))
        if lpips is not None:
            add("lpips", lpips(out.rgb, gt))
        if dynamic_ids:
            dyn = actor_mask(scene, index, dynamic_ids, margin=0.0).to(device) & eval_mask
            add("psnr_dynamic", M.psnr(out.rgb, gt, dyn))

        lidar = lidar_depth_image(scene, index).to(device)
        for key, value in M.depth_errors(out.depth, lidar, eval_mask & (out.alpha > 0.5)).items():
            add(key, value)

        labels = scene.semantics(index)
        pred_labels = None
        if out.semantics is not None:
            pred_labels = viz.semantic_prediction(out.semantics, out.alpha)
            if labels is not None:
                target = labels.to(device).clone()
                target[~eval_mask] = IGNORE_INDEX
                conf += M.confusion_matrix(pred_labels, target, NUM_CLASSES, IGNORE_INDEX)

        if out_dir is not None:
            panels = [viz.to_uint8(gt), viz.to_uint8(out.rgb), viz.colorize_depth(out.depth)]
            if pred_labels is not None:
                panels.append(viz.colorize_labels(pred_labels))
            viz.save_png(viz.hstack(*panels), out_dir / f"{index:05d}_{rec.camera}.png")

    results = {k: float(sum(v) / len(v)) for k, v in rows.items()}
    if conf.sum() > 0:
        results["miou"], _ = M.miou_from_confusion(conf)
    results["num_images"] = len(indices)
    results["num_gaussians"] = len(model.gaussians)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results
