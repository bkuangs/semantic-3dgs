"""Command-line entry point: ``s3dgs {synth,check,segment,train,eval,render}``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def _cmd_synth(args) -> None:
    from semantic_3dgs.data.synthetic import generate_synthetic_scene

    path = generate_synthetic_scene(args.out, num_frames=args.frames, device=args.device)
    print(f"wrote synthetic scene to {path}")


def _cmd_check(args) -> None:
    """Milestone 0 sanity check: LiDAR and boxes projected on images."""
    from semantic_3dgs import viz
    from semantic_3dgs.config import DataConfig
    from semantic_3dgs.data.depth import lidar_depth_image
    from semantic_3dgs.data.masks import actor_mask
    from semantic_3dgs.data.scene import DrivingScene

    scene = DrivingScene(DataConfig(scene_dir=Path(args.scene), cameras=tuple(args.cameras),
                                    downscale=args.downscale, cache_images=False))
    out = Path(args.out)
    frames = args.frames or [0, scene.num_frames // 2, scene.num_frames - 1]
    for rec in scene.images:
        if rec.frame not in frames:
            continue
        image = scene.image(rec.index)
        panel = viz.overlay_points(image, lidar_depth_image(scene, rec.index))
        panel = viz.overlay_mask(panel, actor_mask(scene, rec.index, list(scene.tracks), 0.0))
        viz.save_png(panel, out / f"frame{rec.frame:04d}_{rec.camera}.png")
    print(f"wrote overlays for frames {frames} to {out}")


def _cmd_segment(args) -> None:
    from semantic_3dgs.semantics.segment import segment_scene

    n = segment_scene(args.scene, args.model, tuple(args.cameras) if args.cameras else None,
                      device=args.device, overwrite=args.overwrite)
    print(f"wrote {n} semantic maps")


def _cmd_train(args) -> None:
    from semantic_3dgs.config import load_config
    from semantic_3dgs.training.trainer import Trainer

    cfg = load_config(args.config, args.set)
    metrics = Trainer(cfg).train()
    print(json.dumps(metrics, indent=2))


def _load(run: str, device: str):
    from semantic_3dgs.data.scene import DrivingScene
    from semantic_3dgs.model.scene_model import SceneModel

    path = Path(run)
    if path.is_dir():
        candidates = sorted((path / "checkpoints").glob("step_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no checkpoints in {path / 'checkpoints'}")
        path = candidates[-1]
    model, cfg, _ = SceneModel.load(path, device)
    return model, cfg, DrivingScene(cfg.data), path


def _cmd_eval(args) -> None:
    from semantic_3dgs.data.splits import split_images
    from semantic_3dgs.eval.evaluate import evaluate

    model, cfg, scene, path = _load(args.run, args.device)
    _, test = split_images(scene, cfg.data.holdout_every)
    out = path.parent.parent / "eval" / path.stem
    metrics = evaluate(model, scene, test, cfg, out, use_lpips=args.lpips)
    print(json.dumps(metrics, indent=2))


def _cmd_render(args) -> None:
    from semantic_3dgs.counterfactual import render_counterfactual

    model, cfg, scene, path = _load(args.run, args.device)
    out = Path(args.out) if args.out else path.parent.parent / "renders" / "counterfactual.mp4"
    render_counterfactual(
        model, scene, cfg, out, camera=args.camera, ego_left=args.ego_left,
        track_id=args.track, actor_left=args.actor_left, actor_start=args.actor_start,
        actor_duration=args.actor_duration,
    )
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s3dgs")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="generate the procedural debug scene")
    p.add_argument("--out", default="data/scenes/synthetic")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_synth)

    p = sub.add_parser("check", help="M0: overlay LiDAR depth and boxes on images")
    p.add_argument("scene")
    p.add_argument("--out", default="outputs/check")
    p.add_argument("--cameras", nargs="+", default=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"])
    p.add_argument("--frames", nargs="*", type=int)
    p.add_argument("--downscale", type=int, default=2)
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("segment", help="pseudo-GT 2D semantics with a pretrained network")
    p.add_argument("scene")
    p.add_argument("--model", default="facebook/mask2former-swin-large-mapillary-vistas-semantic")
    p.add_argument("--cameras", nargs="*")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_segment)

    p = sub.add_parser("train")
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("run", help="run directory or checkpoint path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lpips", action="store_true")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("render", help="M5: counterfactual trajectory video")
    p.add_argument("run")
    p.add_argument("--out")
    p.add_argument("--camera", default="FRONT")
    p.add_argument("--ego-left", type=float, default=0.0, help="lateral ego shift (m, +left)")
    p.add_argument("--track", help="track id of the actor to edit")
    p.add_argument("--actor-left", type=float, default=0.0)
    p.add_argument("--actor-start", type=int, default=0)
    p.add_argument("--actor-duration", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_render)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
