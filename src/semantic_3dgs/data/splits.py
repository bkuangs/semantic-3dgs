"""Train / held-out split by frame."""

from __future__ import annotations

from semantic_3dgs.data.scene import DrivingScene


def split_images(scene: DrivingScene, holdout_every: int) -> tuple[list[int], list[int]]:
    """Hold out every image of every ``holdout_every``-th frame (offset by half a period so the
    first and last frames, which are poorly constrained, stay in training)."""
    if holdout_every <= 0:
        return [r.index for r in scene.images], []
    offset = holdout_every // 2
    test = [r.index for r in scene.images if r.frame % holdout_every == offset]
    held = set(test)
    train = [r.index for r in scene.images if r.index not in held]
    return train, test
