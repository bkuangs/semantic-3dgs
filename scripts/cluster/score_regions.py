"""Held-out PSNR / SSIM / LPIPS split into moving-vehicle and background pixels, comparable across models.

Usage: python scripts/cluster/score_regions.py RUN_DIR [RUN_DIR ...] [--checkpoint step_030000.pt]

The built-in evaluation masks each model's own "unmodelled actors", so a static model (every
moving car unmodelled) is scored without car pixels and gets no dynamic PSNR. Here the regions
come from the scene, not the model, so every run is scored on the same pixels:

- moving:     2D boxes of moving vehicles (``init.dynamic_classes``), no margin
- background: everything outside the moving-vehicle boxes grown by ``init.box_margin``
- all:        every valid pixel
Other moving actors (pedestrians, cyclists) are unmodelled by every model and excluded from all
three. Each metric is computed per image inside the region, then averaged over images:
PSNR over the region's pixels, SSIM as the mean of the per-pixel SSIM map (11 px window), and
LPIPS (AlexNet) as the mean of its spatial distance map, whose receptive field reaches past the
region border. Higher is better for PSNR and SSIM, lower for LPIPS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from semantic_3dgs.data.masks import actor_mask
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.data.splits import split_images
from semantic_3dgs.eval import metrics as M
from semantic_3dgs.eval.evaluate import nearest_training_image
from semantic_3dgs.init.lidar import classify_dynamic_tracks
from semantic_3dgs.model.scene_model import SceneModel
from semantic_3dgs.render.pipeline import render, view_from_image

REGIONS = ("all", "moving", "background")
METRICS = ("psnr", "ssim", "lpips")


class SpatialLPIPS:
    def __init__(self, device: str):
        import lpips

        self.net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = pred.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1
        y = target.permute(2, 0, 1)[None] * 2 - 1
        return self.net(x, y)[0, 0]


@torch.no_grad()
def score(run: Path, checkpoint: str | None, device: str, lpips_net: SpatialLPIPS) -> dict:
    ckpts = sorted((run / "checkpoints").glob("step_*.pt"))
    path = run / "checkpoints" / checkpoint if checkpoint else ckpts[-1]
    model, cfg, _ = SceneModel.load(path, device)
    scene = DrivingScene(cfg.data)
    icfg = cfg.init
    moving = classify_dynamic_tracks(scene, icfg.dynamic_classes, icfg.dynamic_displacement_threshold)
    every = classify_dynamic_tracks(
        scene, tuple(sorted({t.category for t in scene.tracks.values()})),
        icfg.dynamic_displacement_threshold,
    )
    others = sorted(set(every) - set(moving))
    train, test = split_images(scene, cfg.data.holdout_every)
    rows: dict[str, dict[str, dict[str, list[float]]]] = {}
    for index in test:
        rec = scene.images[index]
        view = view_from_image(scene, index)
        view.image_index = nearest_training_image(scene, index, train)
        out = render(model, view, cfg.render)
        gt = scene.image(index).to(device)
        ssim = M.ssim_map(out.rgb.clamp(0, 1), gt)
        lp = lpips_net(out.rgb, gt)
        base = scene.valid_mask(rec.camera).to(device)
        if others:
            base = base & ~actor_mask(scene, index, others).to(device)
        masks = {
            "all": base,
            "moving": base & actor_mask(scene, index, moving, margin=0.0).to(device),
            "background": base & ~actor_mask(scene, index, moving, margin=icfg.box_margin).to(device),
        }
        for region, mask in masks.items():
            if mask.sum() < 100:
                continue
            values = {"psnr": M.psnr(out.rgb, gt, mask), "ssim": float(ssim[mask].mean()),
                      "lpips": float(lp[mask].mean())}
            for key in (rec.camera, "mean"):
                for metric, value in values.items():
                    rows.setdefault(key, {}).setdefault(region, {}).setdefault(metric, []).append(value)
    return {"checkpoint": path.name, "num_gaussians": len(model.gaussians),
            "rows": {k: {r: {m: (sum(v) / len(v), len(v)) for m, v in md.items()} for r, md in d.items()}
                     for k, d in rows.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    lpips_net = SpatialLPIPS(args.device)
    fmt = {"psnr": "{:>8.2f}", "ssim": "{:>8.4f}", "lpips": "{:>8.4f}"}
    for run in args.runs:
        r = score(run, args.checkpoint, args.device, lpips_net)
        print(f"\n{run.name}  ({r['checkpoint']}, {r['num_gaussians']:,} Gaussians)")
        print(f"  {'':<14}" + "".join(f"{x:>34}" for x in REGIONS))
        print(f"  {'camera':<14}" + "".join("".join(f"{m:>8}" for m in METRICS) + f"{'n':>10}"
                                            for _ in REGIONS))
        for key, d in sorted(r["rows"].items(), key=lambda kv: kv[0] == "mean"):
            cells = ""
            for x in REGIONS:
                if x in d:
                    cells += "".join(fmt[m].format(d[x][m][0]) for m in METRICS) + f"{d[x]['psnr'][1]:>10}"
                else:
                    cells += f"{'-':>34}"
            print(f"  {key:<14}{cells}")


if __name__ == "__main__":
    main()
