"""``s3dgs`` with extra per-group Gaussian statistics written to ``<output_dir>/diag.jsonl``.

Usage: python scripts/cluster/diag_train.py train CONFIG --set ...   (same arguments as ``s3dgs``)
Every 100 steps logs, separately for background (node 0) and actor Gaussians: count, largest-axis
scale percentiles, fraction with scale > 1 m, mean opacity and fraction below MCMC's 0.005.
"""

from __future__ import annotations

import json
import os

import torch

from semantic_3dgs import cli
from semantic_3dgs.training.trainer import Trainer

_train_step = Trainer.train_step
_STOP = int(os.environ.get("DIAG_STOP_STEP", "0"))  # exit after this step (keeps the full LR schedule)


def _stats(scale: torch.Tensor, opacity: torch.Tensor) -> dict[str, float]:
    if scale.numel() == 0:
        return {"n": 0}
    q = torch.quantile(scale[torch.randperm(len(scale), device=scale.device)[:1_000_000]],
                       torch.tensor([0.5, 0.99], device=scale.device))
    return {
        "n": int(scale.numel()),
        "scale_med": float(q[0]), "scale_p99": float(q[1]), "scale_max": float(scale.max()),
        "frac_scale_gt1m": float((scale > 1.0).float().mean()),
        "op_mean": float(opacity.mean()), "frac_dead": float((opacity <= 0.005).float().mean()),
    }


def train_step(self: Trainer, step: int, index: int) -> dict[str, float]:
    losses = _train_step(self, step, index)
    if step % 100 == 0:
        with torch.no_grad():
            g = self.model.gaussians
            scale, opacity = g.scales.max(dim=1).values, g.opacities.flatten()
            bg = g.node_ids.flatten() == 0
            rec = {"step": step, "bg": _stats(scale[bg], opacity[bg]),
                   "actor": _stats(scale[~bg], opacity[~bg]),
                   "finite": bool(torch.isfinite(g.params["means"]).all())}
        with open(self.out_dir / "diag.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    if _STOP and step >= _STOP:
        raise SystemExit(0)
    return losses


Trainer.train_step = train_step

if __name__ == "__main__":
    cli.main()
