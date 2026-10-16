"""End-to-end milestone checks on a small synthetic scene (CUDA, ~1 min each).

Run with ``pytest tests/student/test_e2e_synthetic.py -m cuda``. Thresholds are set well below
what a correct implementation reaches, so failing one means something is genuinely broken.
Inspect ``<tmp>/run/eval/*`` panels when a check fails.
"""

from pathlib import Path

import pytest
from conftest import cuda

from semantic_3dgs.config import load_config

pytestmark = [cuda, pytest.mark.cuda]

ITERATIONS = 1500
CONFIGS = Path(__file__).parents[2] / "configs" / "synthetic"
# A reference implementation reaches ~35 dB PSNR, ~33 dB dynamic PSNR, 0.011 AbsRel, 0.98 mIoU.
THRESHOLDS = {
    "m1_psnr": 30.0,
    "m2_psnr": 30.0,
    "m2_psnr_dynamic": 28.0,
    "m3_psnr": 30.0,
    "m3_depth_abs_rel": 0.05,
    "m4_miou": 0.85,
}


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    from semantic_3dgs.data.synthetic import generate_synthetic_scene

    return generate_synthetic_scene(tmp_path_factory.mktemp("synth"), num_frames=24, width=320, height=208)


def _train(config: str, root, tmp_path) -> dict[str, float]:
    from semantic_3dgs.training.trainer import Trainer

    cfg = load_config(CONFIGS / f"{config}.yaml", [
        f"data.scene_dir={root}", f"output_dir={tmp_path / 'run'}", f"train.iterations={ITERATIONS}",
        "train.refine_stop=1200", "train.cap_max=300000", "data.holdout_every=6",
        f"train.eval_every={ITERATIONS}", f"train.checkpoint_every={ITERATIONS}",
    ])
    return Trainer(cfg).train()


def test_m1_static_background_reconstructs(synthetic_root, tmp_path):
    m = _train("m1_static", synthetic_root, tmp_path)
    assert m["psnr"] > THRESHOLDS["m1_psnr"]


def test_m2_dynamic_car_is_sharp(synthetic_root, tmp_path):
    m = _train("m2_dynamic", synthetic_root, tmp_path)
    assert m["psnr"] > THRESHOLDS["m2_psnr"]
    assert m["psnr_dynamic"] > THRESHOLDS["m2_psnr_dynamic"]


def test_m3_depth_and_robustness(synthetic_root, tmp_path):
    m = _train("m3_full", synthetic_root, tmp_path)
    assert m["psnr"] > THRESHOLDS["m3_psnr"]
    assert m["depth_abs_rel"] < THRESHOLDS["m3_depth_abs_rel"]


def test_m4_semantics_are_distilled(synthetic_root, tmp_path):
    m = _train("m4_semantic", synthetic_root, tmp_path)
    assert m["miou"] > THRESHOLDS["m4_miou"]
