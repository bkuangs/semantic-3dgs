from pathlib import Path

import pytest

from semantic_3dgs.config import config_from_dict, config_to_dict, load_config

CONFIGS = sorted((Path(__file__).parents[2] / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(p.parents[1])))
def test_every_shipped_config_loads(path):
    if path.name == "base.yaml":
        pytest.skip("base configs are only inherited")
    cfg = load_config(path)
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_overrides_and_types(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  scene_dir: some/where\n  frame_range: [0, 50]\n")
    cfg = load_config(path, ["train.iterations=10", "features.dynamic=true", "render.background=[1,1,1]"])
    assert cfg.train.iterations == 10
    assert cfg.features.dynamic is True
    assert cfg.data.scene_dir == Path("some/where")
    assert cfg.data.frame_range == (0, 50)
    assert cfg.render.background == (1.0, 1.0, 1.0)


def test_inherit(tmp_path):
    (tmp_path / "base.yaml").write_text("train:\n  iterations: 5\n  seed: 3\n")
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\ntrain:\n  iterations: 7\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert (cfg.train.iterations, cfg.train.seed) == (7, 3)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="iteratons"):
        config_from_dict({"train": {"iteratons": 3}})
