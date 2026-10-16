"""Driving semantic taxonomy and mappings from off-the-shelf segmentation label spaces."""

from __future__ import annotations

import numpy as np

CLASSES: tuple[str, ...] = (
    "road",
    "lane_marking",
    "sidewalk",
    "building",
    "vegetation",
    "sky",
    "vehicle",
    "pedestrian",
    "cyclist",
    "traffic_infra",
    "other",
)
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
SKY = CLASS_INDEX["sky"]

PALETTE = np.array(
    [
        [128, 64, 128],
        [255, 255, 255],
        [244, 35, 232],
        [70, 70, 70],
        [107, 142, 35],
        [70, 130, 180],
        [0, 0, 142],
        [220, 20, 60],
        [119, 11, 32],
        [220, 220, 0],
        [120, 120, 120],
    ],
    dtype=np.uint8,
)

# Keyword rules applied to lower-cased source label names (Cityscapes, Mapillary Vistas, ...).
# First match wins, so more specific keywords come first.
_KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    ("ego vehicle", "__ignore__"),
    ("car mount", "__ignore__"),
    ("unlabeled", "__ignore__"),
    ("lane marking", "lane_marking"),
    ("marker", "lane_marking"),
    ("marking", "lane_marking"),
    ("crosswalk", "lane_marking"),
    ("sidewalk", "sidewalk"),
    ("curb", "sidewalk"),
    ("road", "road"),
    ("bike lane", "road"),
    ("service lane", "road"),
    ("manhole", "road"),
    ("pothole", "road"),
    ("parking", "road"),
    ("rail track", "road"),
    ("sky", "sky"),
    ("vegetation", "vegetation"),
    ("terrain", "vegetation"),
    ("building", "building"),
    ("wall", "building"),
    ("fence", "building"),
    ("bridge", "building"),
    ("tunnel", "building"),
    ("bicyclist", "cyclist"),
    ("motorcyclist", "cyclist"),
    ("rider", "cyclist"),
    ("bicycle", "cyclist"),
    ("motorcycle", "cyclist"),
    ("person", "pedestrian"),
    ("pedestrian", "pedestrian"),
    ("car", "vehicle"),
    ("truck", "vehicle"),
    ("bus", "vehicle"),
    ("train", "vehicle"),
    ("trailer", "vehicle"),
    ("caravan", "vehicle"),
    ("vehicle", "vehicle"),
    ("traffic", "traffic_infra"),
    ("sign", "traffic_infra"),
    ("pole", "traffic_infra"),
    ("street light", "traffic_infra"),
)


def map_label_name(name: str) -> int:
    lowered = name.lower().replace("_", " ").replace("-", " ")
    for keyword, target in _KEYWORD_RULES:
        if keyword in lowered:
            return IGNORE_INDEX if target == "__ignore__" else CLASS_INDEX[target]
    return CLASS_INDEX["other"]


def build_lookup(id2label: dict[int, str]) -> np.ndarray:
    """Lookup table from a model's label ids to this taxonomy (unmapped ids -> IGNORE)."""
    lut = np.full(max(id2label) + 1, IGNORE_INDEX, dtype=np.uint8)
    for idx, name in id2label.items():
        lut[idx] = map_label_name(name)
    return lut
