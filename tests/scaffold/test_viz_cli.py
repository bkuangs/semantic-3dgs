import numpy as np
import torch

from semantic_3dgs import viz
from semantic_3dgs.classes import IGNORE_INDEX, SKY
from semantic_3dgs.cli import build_parser


def test_colorize_and_prediction():
    labels = torch.tensor([[0, 1], [IGNORE_INDEX, 5]])
    rgb = viz.colorize_labels(labels)
    assert rgb.shape == (2, 2, 3) and rgb[1, 0].tolist() == [0, 0, 0]
    pred = viz.semantic_prediction(torch.randn(2, 2, 11), torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
    assert pred[0, 1] == SKY
    assert viz.colorize_depth(torch.rand(3, 4) * 80).dtype == np.uint8


def test_write_video(tmp_path):
    frames = [np.full((33, 47, 3), i * 20, dtype=np.uint8) for i in range(5)]
    viz.write_video(frames, tmp_path / "v.mp4", fps=5)
    assert (tmp_path / "v.mp4").stat().st_size > 0


def test_cli_parses():
    parser = build_parser()
    args = parser.parse_args(["train", "configs/synthetic/m1_static.yaml", "--set", "train.iterations=5"])
    assert args.set == ["train.iterations=5"]
    args = parser.parse_args(["render", "outputs/x", "--ego-left", "2", "--track", "abc"])
    assert args.ego_left == 2.0 and args.track == "abc"
