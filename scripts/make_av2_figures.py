"""Build the Argoverse 2 README figures in ``docs/images`` from a trained run's eval renders.

CPU only: it just crops and lays out the ``eval/step_*/{index}_{CAMERA}.png`` strips that
``s3dgs train`` writes (camera | render | depth | semantics), so it runs on a login node::

    uv run python scripts/make_av2_figures.py ~/scratch/s3dgs/outputs/av2_full/D_budget6M_n210
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from semantic_3dgs.classes import CLASSES, PALETTE

PANELS = ("Camera (held out)", "Gaussian render", "Rendered depth", "Rendered semantics")
RING = ("SIDE_LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_RIGHT")


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def split_panels(path: Path) -> list[Image.Image]:
    strip = Image.open(path).convert("RGB")
    n = 4 if strip.width % 4 == 0 else 3
    w = strip.width // n
    return [strip.crop((i * w, 0, (i + 1) * w, strip.height)) for i in range(n)]


def labelled(path: Path) -> list[tuple[Image.Image, str]]:
    panels = split_panels(path)
    return list(zip(panels, PANELS[: len(panels)], strict=True))


def resize_h(im: Image.Image, h: int) -> Image.Image:
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def caption(im: Image.Image, text: str, size: int = 15) -> Image.Image:
    bar = size + 10
    out = Image.new("RGB", (im.width, im.height + bar), (24, 24, 24))
    ImageDraw.Draw(out).text((6, 4), text, fill=(235, 235, 235), font=_font(size))
    out.paste(im, (0, bar))
    return out


def hcat(images: list[Image.Image], pad: int = 4) -> Image.Image:
    h = max(im.height for im in images)
    out = Image.new("RGB", (sum(im.width for im in images) + pad * (len(images) - 1), h), "white")
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + pad
    return out


def vcat(images: list[Image.Image], pad: int = 4) -> Image.Image:
    w = max(im.width for im in images)
    out = Image.new("RGB", (w, sum(im.height for im in images) + pad * (len(images) - 1)), "white")
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height + pad
    return out


def legend(width: int, size: int = 13) -> Image.Image:
    font = _font(size)
    sw, gap, x, y, rows = size + 2, 14, 6, 6, [[]]
    items = [(n.replace("_", " "), tuple(int(c) for c in PALETTE[i])) for i, n in enumerate(CLASSES)]
    for name, colour in items:
        w = sw + 4 + int(font.getlength(name)) + gap
        if x + w > width and rows[-1]:
            rows.append([])
            x = 6
        rows[-1].append((x, name, colour))
        x += w
    out = Image.new("RGB", (width, len(rows) * (sw + 6) + 6), (24, 24, 24))
    d = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        y = 6 + r * (sw + 6)
        for x, name, colour in row:
            d.rectangle((x, y, x + sw, y + sw), fill=colour, outline=(200, 200, 200))
            d.text((x + sw + 4, y), name, fill=(235, 235, 235), font=font)
    return out


def index_of(path: Path) -> int:
    return int(re.match(r"(\d+)_", path.name).group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path, help="training output dir containing eval/step_*")
    ap.add_argument("--step", default=None, help="eval step dir name (default: latest)")
    ap.add_argument("--hero-index", type=int, default=375, help="FRONT image index for the hero")
    ap.add_argument("--out", type=Path, default=Path("docs/images"))
    args = ap.parse_args()

    steps = sorted((args.run / "eval").glob("step_*"))
    step = args.run / "eval" / args.step if args.step else steps[-1]
    args.out.mkdir(parents=True, exist_ok=True)
    fronts = sorted(step.glob("*_FRONT.png"), key=index_of)

    # Hero: one held-out front view, all four outputs side by side.
    hero = step / f"{args.hero_index:05d}_FRONT.png"
    panels = [caption(resize_h(p, 420), t) for p, t in labelled(hero)]
    row = hcat(panels)
    vcat([row, legend(row.width)], pad=0).save(args.out / "av2_hero.png", optimize=True)

    # Drive: every held-out front view along the log, as a looping GIF.
    frames = []
    for path in fronts:
        panels = [caption(resize_h(p, 300), t, 13) for p, t in labelled(path)]
        frames.append(hcat(panels))
    frames = [f.quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(
        args.out / "av2_drive.gif", save_all=True, append_images=frames[1:], duration=700, loop=0, optimize=True
    )

    # Ring: the five cameras of the hero frame, real on top, rendered below.
    # The images of one frame have consecutive indices, starting with FRONT.
    by_cam = {
        p.stem.split("_", 1)[1]: p
        for p in step.glob("*.png")
        if args.hero_index <= index_of(p) < args.hero_index + len(RING)
    }
    cams = [c for c in RING if c in by_cam]
    real = [caption(resize_h(split_panels(by_cam[c])[0], 220), c.replace("_", " ").lower(), 13) for c in cams]
    fake = [resize_h(split_panels(by_cam[c])[1], 220) for c in cams]
    vcat([hcat(real), hcat(fake)]).save(args.out / "av2_cameras.png", optimize=True)
    print(f"wrote av2_hero.png, av2_drive.gif ({len(frames)} frames), av2_cameras.png from {step}")


if __name__ == "__main__":
    main()
