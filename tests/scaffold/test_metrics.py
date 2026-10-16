import torch

from semantic_3dgs.eval import metrics as M


def test_psnr():
    a = torch.zeros(4, 4, 3)
    b = torch.full((4, 4, 3), 0.1)
    assert abs(M.psnr(a, b) - 20.0) < 1e-4
    mask = torch.zeros(4, 4, dtype=torch.bool)
    assert M.psnr(a, b, mask) != M.psnr(a, b, mask)  # NaN for empty mask


def test_ssim_identity_and_sensitivity():
    img = torch.rand(32, 32, 3)
    assert abs(M.ssim(img, img) - 1.0) < 1e-5
    assert M.ssim(img, torch.rand(32, 32, 3)) < 0.5
    assert M.ssim_map(img, img).shape == (32, 32)


def test_miou():
    target = torch.tensor([0, 0, 1, 1, 255])
    pred = torch.tensor([0, 1, 1, 1, 0])
    conf = M.confusion_matrix(pred, target, 3, 255)
    miou, iou = M.miou_from_confusion(conf)
    assert abs(iou[0] - 0.5) < 1e-6 and abs(iou[1] - 2 / 3) < 1e-6 and iou[2].isnan()
    assert abs(miou - (0.5 + 2 / 3) / 2) < 1e-6


def test_depth_errors():
    lidar = torch.tensor([[0.0, 10.0], [20.0, 0.0]])
    pred = torch.tensor([[5.0, 11.0], [18.0, 1.0]])
    out = M.depth_errors(pred, lidar)
    assert abs(out["depth_abs_rel"] - 0.1) < 1e-6
