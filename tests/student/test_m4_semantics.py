"""Milestone 4: semantic distillation."""

import torch
import torch.nn.functional as F

from semantic_3dgs.classes import IGNORE_INDEX
from semantic_3dgs.losses import semantic_loss


def test_semantic_loss_matches_cross_entropy_on_valid_pixels():
    logits = torch.randn(6, 7, 5, requires_grad=True)
    labels = torch.randint(0, 5, (6, 7))
    labels[0] = IGNORE_INDEX
    mask = torch.ones(6, 7, dtype=torch.bool)
    mask[:, :2] = False
    loss = semantic_loss(logits, labels, mask)
    valid = mask & (labels != IGNORE_INDEX)
    expected = F.cross_entropy(logits[valid], labels[valid])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[:, :2].abs().sum() == 0
    assert logits.grad[0].abs().sum() == 0


def test_semantic_loss_without_valid_pixels_is_zero_with_graph():
    logits = torch.randn(3, 3, 4, requires_grad=True)
    labels = torch.full((3, 3), IGNORE_INDEX)
    loss = semantic_loss(logits, labels, torch.ones(3, 3, dtype=torch.bool))
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
