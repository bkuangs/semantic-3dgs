"""Pseudo ground-truth 2D semantics from an off-the-shelf segmentation network.

Defaults to Mask2Former trained on Mapillary Vistas, because Cityscapes has no lane-marking class.
Labels are mapped to :mod:`semantic_3dgs.classes` by name and written as uint8 PNGs next to the
images (``semantics/{CAMERA}/{frame:04d}.png``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from semantic_3dgs.classes import build_lookup

DEFAULT_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
CITYSCAPES_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


@torch.no_grad()
def segment_scene(
    scene_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    cameras: tuple[str, ...] | None = None,
    device: str = "cuda",
    max_side: int = 1024,
    overwrite: bool = False,
) -> int:
    from transformers import (
        AutoImageProcessor,
        AutoModelForSemanticSegmentation,
        AutoModelForUniversalSegmentation,
    )

    root = Path(scene_dir)
    meta = json.loads((root / "meta.json").read_text())
    processor = AutoImageProcessor.from_pretrained(model_name)
    universal = "mask2former" in model_name or "oneformer" in model_name
    loader = AutoModelForUniversalSegmentation if universal else AutoModelForSemanticSegmentation
    model = loader.from_pretrained(model_name).to(device).eval()
    lut = build_lookup({int(k): v for k, v in model.config.id2label.items()})

    jobs = [
        (frame["index"], cam, entry["path"])
        for frame in meta["frames"]
        for cam, entry in frame["images"].items()
        if cameras is None or cam in cameras
    ]
    written = 0
    for index, cam, rel in tqdm(jobs, desc="segment"):
        out_path = root / "semantics" / cam / f"{index:04d}.png"
        if out_path.exists() and not overwrite:
            continue
        image = Image.open(root / rel).convert("RGB")
        w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        small = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        inputs = processor(images=small, return_tensors="pt").to(device)
        outputs = model(**inputs)
        labels = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
        mapped = lut[labels.cpu().numpy().clip(0, len(lut) - 1)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mapped.astype(np.uint8)).save(out_path)
        written += 1
    return written
