"""Summarise diagnostic runs: losses from train_log.jsonl plus per-group stats from diag.jsonl.

Usage: python scripts/cluster/diag_report.py OUTPUT_ROOT RUN [RUN ...] [--every 1000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2] if values else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=1000)
    args = ap.parse_args()
    for run in args.runs:
        log = load(args.root / run / "train_log.jsonl")
        diag = {r["step"]: r for r in load(args.root / run / "diag.jsonl")}
        last = log[-1]["step"] if log else -1
        print(f"\n== {run}  (last step {last}, n={log[-1]['num_gaussians'] if log else '?'})")
        print(f"{'steps':>11} {'rgb':>6} {'depth':>6} {'op_reg':>7} {'sc_reg':>7} | "
              f"{'bg med':>7} {'bg p99':>7} {'bg>1m':>6} {'bg op':>6} {'bg dead':>7} | "
              f"{'ac med':>7} {'ac p99':>7} {'ac>1m':>6} {'ac op':>6} {'ac dead':>7}")
        for a in range(0, last + 1, args.every):
            w = [r for r in log if a <= r["step"] < a + args.every]
            d = diag.get(a + args.every - 100) or diag.get(a)
            row = (f"{a:>5}-{a + args.every:<5} "
                   + " ".join(f"{median([r[k] for r in w]):{n}.3f}"
                              for k, n in (("rgb", 6), ("depth", 6), ("opacity_reg", 7),
                                           ("scale_reg", 7))))
            for g in ("bg", "actor"):
                s = d[g] if d else {}
                row += " | " + " ".join(
                    f"{s.get(k, float('nan')):{n}.3f}" for k, n in
                    (("scale_med", 7), ("scale_p99", 7), ("frac_scale_gt1m", 6), ("op_mean", 6),
                     ("frac_dead", 7)))
            print(row)


if __name__ == "__main__":
    main()
