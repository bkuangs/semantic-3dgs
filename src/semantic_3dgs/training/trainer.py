"""Training loop (implemented). Every theory piece it calls lives in a stub elsewhere."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from semantic_3dgs.config import ExperimentConfig, config_to_dict
from semantic_3dgs.data.depth import lidar_depth_image
from semantic_3dgs.data.masks import actor_mask
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.data.splits import split_images
from semantic_3dgs.eval.evaluate import evaluate
from semantic_3dgs.losses import Batch, compute_losses
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.render.pipeline import render, view_from_image

log = logging.getLogger(__name__)

_GAUSSIAN_LR_FIELDS = {
    "means": "lr_means",
    "scales": "lr_scales",
    "quats": "lr_quats",
    "opacities": "lr_opacities",
    "sh0": "lr_sh0",
    "shN": "lr_shN",
    "semantics": "lr_semantics",
}


def make_strategy(cfg: ExperimentConfig):
    from gsplat import DefaultStrategy, MCMCStrategy

    t = cfg.train
    if t.strategy == "mcmc":
        return MCMCStrategy(
            cap_max=t.cap_max,
            noise_lr=t.noise_lr,
            refine_start_iter=t.refine_start,
            refine_stop_iter=t.refine_stop,
            refine_every=t.refine_every,
        )
    if t.strategy == "default":
        return DefaultStrategy(
            grow_grad2d=t.densify_grad_threshold,
            refine_start_iter=t.refine_start,
            refine_stop_iter=t.refine_stop,
            refine_every=t.refine_every,
        )
    raise ValueError(f"unknown strategy {t.strategy!r}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, cfg: ExperimentConfig, scene: DrivingScene | None = None):
        self.cfg = cfg
        seed_everything(cfg.train.seed)
        self.device = torch.device(cfg.train.device)
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg)))

        self.scene = scene or DrivingScene(cfg.data)
        self.train_ids, self.test_ids = split_images(self.scene, cfg.data.holdout_every)
        self.extent = self.scene.scene_extent()
        log.info(
            "scene %s: %d frames, %d train / %d test images, extent %.1f m",
            self.scene.segment, self.scene.num_frames, len(self.train_ids),
            len(self.test_ids), self.extent,
        )

        self.model = SceneModel.build(self.scene, cfg).to(self.device)
        log.info(
            "model: %d Gaussians, %d nodes, %d unmodelled actors",
            len(self.model.gaussians), self.model.graph.num_nodes,
            len(self.model.unmodelled_track_ids),
        )
        self.optimizers = self._gaussian_optimizers()
        self.aux_optimizer = self._aux_optimizer()
        gamma = cfg.train.lr_means_final_factor ** (1.0 / max(cfg.train.iterations, 1))
        self.means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=gamma
        )
        self.strategy = make_strategy(cfg)
        self.strategy.check_sanity(self.model.gaussians.params, self.optimizers)
        if cfg.train.strategy == "mcmc":
            self.strategy_state = self.strategy.initialize_state()
        else:
            self.strategy_state = self.strategy.initialize_state(scene_scale=self.extent)
        self._depth_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._mask_cache: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ setup
    def _gaussian_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        t = self.cfg.train
        params = self.model.gaussians.params
        optimizers = {}
        for name in self.model.gaussians.trainable_names:
            lr = getattr(t, _GAUSSIAN_LR_FIELDS[name])
            if name == "means":
                lr *= self.extent
            optimizers[name] = torch.optim.Adam([params[name]], lr=lr, eps=1e-15)
        return optimizers

    def _aux_optimizer(self) -> torch.optim.Optimizer | None:
        t = self.cfg.train
        groups = []
        if self.model.sky is not None:
            groups.append({"params": self.model.sky.parameters(), "lr": t.lr_sky})
        if self.model.exposure is not None:
            groups.append({"params": self.model.exposure.parameters(), "lr": t.lr_exposure})
        if self.model.graph.pose_refiner is not None:
            groups.append(
                {"params": self.model.graph.pose_refiner.parameters(), "lr": t.lr_pose}
            )
        return torch.optim.Adam(groups) if groups else None

    # ------------------------------------------------------------------ data
    def load_batch(self, index: int) -> Batch:
        scene, f = self.scene, self.cfg.features
        rec = scene.images[index]
        mask = scene.valid_mask(rec.camera)
        if f.mask_unmodelled_actors and self.model.unmodelled_track_ids:
            if index not in self._mask_cache:
                self._mask_cache[index] = actor_mask(
                    scene, index, self.model.unmodelled_track_ids, self.cfg.init.box_margin
                )
            mask = mask & ~self._mask_cache[index]
        semantics = scene.semantics(index) if (f.semantics or f.sky) else None
        depth = self._lidar_depth(index) if f.depth_loss else None
        dev = self.device
        return Batch(
            image=scene.image(index).to(dev, non_blocking=True),
            mask=mask.to(dev),
            semantics=semantics.to(dev) if semantics is not None else None,
            lidar_depth=depth.to(dev) if depth is not None else None,
        )

    def _lidar_depth(self, index: int) -> torch.Tensor:
        cam = self.scene.cameras[self.scene.images[index].camera]
        if index not in self._depth_cache:
            dense = lidar_depth_image(self.scene, index).flatten()
            nz = torch.nonzero(dense, as_tuple=True)[0]
            self._depth_cache[index] = (nz.int(), dense[nz].half())
        nz, values = self._depth_cache[index]
        dense = torch.zeros(cam.height * cam.width)
        dense[nz.long()] = values.float()
        return dense.view(cam.height, cam.width)

    # ------------------------------------------------------------------ loop
    def sh_degree_at(self, step: int) -> int:
        return min(step // self.cfg.train.sh_degree_interval, self.cfg.model.sh_degree)

    def train_step(self, step: int, index: int) -> dict[str, float]:
        batch = self.load_batch(index)
        out = render(
            self.model, view_from_image(self.scene, index), self.cfg.render, self.sh_degree_at(step)
        )
        losses = compute_losses(out, batch, self.model, self.cfg)

        params = self.model.gaussians.params
        self.strategy.step_pre_backward(params, self.optimizers, self.strategy_state, step, out.info)
        losses["total"].backward()

        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if self.aux_optimizer is not None:
            self.aux_optimizer.step()
            self.aux_optimizer.zero_grad(set_to_none=True)
        self.means_scheduler.step()

        if self.cfg.train.strategy == "mcmc":
            self.strategy.step_post_backward(
                params, self.optimizers, self.strategy_state, step, out.info,
                lr=self.means_scheduler.get_last_lr()[0],
            )
        else:
            self.strategy.step_post_backward(
                params, self.optimizers, self.strategy_state, step, out.info,
                packed=self.cfg.render.packed,
            )
        return {k: float(v.detach()) for k, v in losses.items()}

    def train(self) -> dict[str, float]:
        from tqdm import trange

        t = self.cfg.train
        log_path = self.out_dir / "train_log.jsonl"
        order: list[int] = []
        start = time.time()
        metrics: dict[str, float] = {}
        bar = trange(t.iterations, desc="train", dynamic_ncols=True)
        for step in bar:
            if not order:
                order = random.sample(self.train_ids, len(self.train_ids))
            losses = self.train_step(step, order.pop())
            if step % t.log_every == 0 or step == t.iterations - 1:
                record = {"step": step, "time": time.time() - start,
                          "num_gaussians": len(self.model.gaussians), **losses}
                with log_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                bar.set_postfix(loss=f"{losses['total']:.4f}", n=len(self.model.gaussians))
            last = step == t.iterations - 1
            if t.eval_every > 0 and self.test_ids and ((step + 1) % t.eval_every == 0 or last):
                metrics = self.evaluate(step + 1)
            if t.checkpoint_every > 0 and ((step + 1) % t.checkpoint_every == 0 or last):
                self.model.save(self.out_dir / "checkpoints" / f"step_{step + 1:06d}.pt",
                                self.cfg, step + 1)
        return metrics

    def evaluate(self, step: int) -> dict[str, float]:
        self.model.eval()
        metrics = evaluate(
            self.model, self.scene, self.test_ids, self.cfg, self.out_dir / "eval" / f"step_{step:06d}"
        )
        self.model.train()
        log.info("eval @%d: %s", step, {k: round(v, 4) for k, v in metrics.items()})
        return metrics
