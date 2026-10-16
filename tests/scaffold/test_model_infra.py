import torch

from semantic_3dgs.config import ExperimentConfig, FeaturesConfig
from semantic_3dgs.init.initialize import knn_scales
from semantic_3dgs.init.lidar import voxel_downsample
from semantic_3dgs.init.objects import fill_box
from semantic_3dgs.model.gaussians import GaussianBuffer
from semantic_3dgs.model.scene_model import SceneModel


def _buffer(n=10, sem=4):
    node_ids = torch.tensor([0] * (n - 3) + [1, 1, 2], dtype=torch.int32)
    return GaussianBuffer.from_points(torch.randn(n, 3), torch.full((n,), 0.1), node_ids,
                                      sh_degree=2, num_semantic_classes=sem)


def test_buffer_layout():
    g = _buffer()
    assert len(g) == 10
    assert g.sh_coeffs.shape == (10, 9, 3)
    assert g.semantics.shape == (10, 4)
    assert not g.node_ids.requires_grad
    assert "node_ids" not in g.trainable_names
    torch.testing.assert_close(g.scales, torch.full((10, 3), 0.1))
    torch.testing.assert_close(g.opacities, torch.full((10,), 0.1))
    assert g.count_per_node(3).tolist() == [7, 2, 1]


def test_gsplat_ops_keep_node_ids_aligned():
    from gsplat.strategy.ops import duplicate, remove

    g = _buffer()
    opts = {k: torch.optim.Adam([g.params[k]]) for k in g.trainable_names}
    before = g.means.detach().clone()
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[0, 8]] = True
    duplicate(g.params, opts, {}, mask)
    assert len(g) == 12 and g.node_ids.tolist()[-2:] == [0, 1]
    torch.testing.assert_close(g.means[-1], before[8])
    keep_out = torch.zeros(12, dtype=torch.bool)
    keep_out[0] = True
    remove(g.params, opts, {}, keep_out)
    assert len(g) == 11 and g.count_per_node(3).tolist() == [7, 3, 1]


def test_voxel_downsample_and_fill():
    pts = torch.tensor([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [1.0, 1.0, 1.0]])
    out = voxel_downsample(pts, 0.05)
    assert len(out) == 2
    assert torch.isclose(out, torch.tensor([0.02, 0.02, 0.02])).all(1).any()
    box = fill_box(torch.tensor([4.0, 2.0, 1.0]), 100)
    assert (box.abs() <= torch.tensor([2.0, 1.0, 0.5])).all()


def test_knn_scales():
    pts = torch.stack(torch.meshgrid(*(torch.arange(4.0),) * 3, indexing="ij"), -1).reshape(-1, 3)
    assert torch.allclose(knn_scales(pts)[21], torch.tensor(1.0))


def test_checkpoint_roundtrip(tmp_path):
    cfg = ExperimentConfig(features=FeaturesConfig(dynamic=True, sky=True, exposure=True,
                                                   pose_refine=True, semantics=True))
    g = _buffer(sem=11)
    poses = torch.eye(4).repeat(3, 5, 1, 1)
    visible = torch.ones(3, 5, dtype=torch.bool)
    model = SceneModel._assemble(cfg, g, [None, "a", "b"], poses, visible, 7, ["ped"])
    with torch.no_grad():
        model.sky.faces.fill_(0.3)
        model.exposure.affine[3, 0, 3] = 0.2
    model.save(tmp_path / "m.pt", cfg, step=11)
    loaded, cfg2, ckpt = SceneModel.load(tmp_path / "m.pt")
    assert cfg2 == cfg and ckpt["step"] == 11
    assert loaded.graph.node_track_ids == [None, "a", "b"]
    assert loaded.unmodelled_track_ids == ["ped"]
    a, b = model.state_dict(), loaded.state_dict()
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], msg=k)
