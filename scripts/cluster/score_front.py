"""PSNR of held-out front-camera images in frames 0-59, from a run's evaluation panels.

Usage: python scripts/cluster/score_front.py RUN_DIR [RUN_DIR ...] [--step N] [--last-frame 59]

Panels are ``eval/step_XXXXXX/{index:05d}_{CAMERA}.png`` (photo | render | ...). Pixels outside
the scene's ``masks/{CAMERA}.png`` are ignored; moving actors are not, so only compare runs
scored with this script against each other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_3dgs.config import load_config
from semantic_3dgs.data.scene import DrivingScene

CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")


def score(run: Path, step: int | None, last_frame: int) -> dict:
    cfg = load_config(run / "config.yaml", ["data.cache_images=false"])
    scene = DrivingScene(cfg.data)
    step_dir = (run / "eval" / f"step_{step:06d}" if step is not None
                else sorted((run / "eval").glob("step_*"))[-1])
    n_panels = 4 if cfg.features.semantics else 3
    psnrs: dict[str, list[float]] = {c: [] for c in CAMERAS}
    for png in sorted(step_dir.glob("*.png")):
        index, camera = png.stem.split("_", 1)
        rec = scene.images[int(index)]
        assert rec.camera == camera, (png, rec.camera)
        if camera not in CAMERAS or rec.frame > last_frame:
            continue
        panel = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64) / 255.0
        w = panel.shape[1] // n_panels
        gt, pred = panel[:, :w], panel[:, w : 2 * w]
        mask = np.ones(gt.shape[:2], bool)
        mask_path = scene.root / "masks" / f"{camera}.png"
        if mask_path.exists():
            m = Image.open(mask_path).convert("L").resize((w, gt.shape[0]), Image.NEAREST)
            mask = np.asarray(m) > 127
        psnrs[camera].append(-10 * np.log10(((gt - pred) ** 2)[mask].mean()))
    return {"step_dir": step_dir.name, "n": {c: len(v) for c, v in psnrs.items()},
            **{c: float(np.mean(v)) if v else float("nan") for c, v in psnrs.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--step", type=int)
    ap.add_argument("--last-frame", type=int, default=59)
    args = ap.parse_args()
    print(f"{'run':<28}{'step':<14}" + "".join(f"{c:>13}" for c in CAMERAS))
    for run in args.runs:
        r = score(run, args.step, args.last_frame)
        print(f"{run.name:<28}{r['step_dir']:<14}" + "".join(f"{r[c]:>13.2f}" for c in CAMERAS)
              + f"   n={r['n']}")


if __name__ == "__main__":
    main()
