"""Flat Gaussian parameter buffer shared by every scene-graph node.

All Gaussians (background and every object) live in one set of tensors; ``node_ids`` says which
node each Gaussian belongs to, and its ``means``/``quats`` are expressed in that node's local
frame. Keeping one flat buffer means gsplat's densification strategies (which clone, split,
prune and relocate rows of every tensor in the dict) work unchanged - ``node_ids`` is a
non-trainable entry, so it is carried along automatically.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


class GaussianBuffer(nn.Module):
    def __init__(self, params: nn.ParameterDict, sh_degree: int):
        super().__init__()
        self.params = params
        self.sh_degree = sh_degree

    @classmethod
    def from_points(
        cls,
        means: Tensor,
        scales: Tensor,
        node_ids: Tensor,
        sh_degree: int = 3,
        initial_opacity: float = 0.1,
        num_semantic_classes: int = 0,
        colors: Tensor | None = None,
    ) -> GaussianBuffer:
        n = len(means)
        if colors is None:
            colors = torch.full((n, 3), 0.5)
        num_rest = (sh_degree + 1) ** 2 - 1
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0
        params = {
            "means": means.float(),
            "scales": torch.log(scales.float())[:, None].repeat(1, 3),
            "quats": quats,
            "opacities": torch.full((n,), math.log(initial_opacity / (1 - initial_opacity))),
            "sh0": rgb_to_sh0(colors.float())[:, None, :],
            "shN": torch.zeros(n, num_rest, 3),
        }
        if num_semantic_classes > 0:
            params["semantics"] = torch.zeros(n, num_semantic_classes)
        pdict = nn.ParameterDict({k: nn.Parameter(v) for k, v in params.items()})
        pdict["node_ids"] = nn.Parameter(node_ids.int(), requires_grad=False)
        return cls(pdict, sh_degree)

    # ------------------------------------------------------------------ views
    def __len__(self) -> int:
        return self.params["means"].shape[0]

    @property
    def means(self) -> Tensor:
        return self.params["means"]

    @property
    def quats(self) -> Tensor:
        return self.params["quats"]

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.params["scales"])

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self.params["opacities"])

    @property
    def sh_coeffs(self) -> Tensor:
        return torch.cat([self.params["sh0"], self.params["shN"]], dim=1)

    @property
    def semantics(self) -> Tensor | None:
        return self.params["semantics"] if "semantics" in self.params else None

    @property
    def node_ids(self) -> Tensor:
        return self.params["node_ids"]

    @property
    def trainable_names(self) -> list[str]:
        return [k for k, p in self.params.items() if p.requires_grad]

    def count_per_node(self, num_nodes: int) -> Tensor:
        return torch.bincount(self.node_ids.long(), minlength=num_nodes)

    # ------------------------------------------------------------------ io
    def allocate_like(self, state_dict: dict[str, Tensor], prefix: str = "") -> None:
        """Resize parameters to the shapes stored in a checkpoint (densification changes N)."""
        full = prefix + "params."
        for key, value in state_dict.items():
            if key.startswith(full):
                name = key[len(full):]
                trainable = value.is_floating_point() and name != "node_ids"
                self.params[name] = nn.Parameter(torch.empty_like(value), requires_grad=trainable)
