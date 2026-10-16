"""Typed experiment configuration loaded from YAML.

Every section is a frozen dataclass. Unknown keys are rejected so typos fail loudly.
Command-line overrides use dotted paths, e.g. ``--set train.iterations=500``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    scene_dir: Path = Path("data/scenes/synthetic")
    cameras: tuple[str, ...] = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
    # Half-open [start, end) frame window; None uses the full segment.
    frame_range: tuple[int, int] | None = None
    downscale: int = 2
    holdout_every: int = 10
    cache_images: bool = True
    semantics_subdir: str = "semantics"


@dataclass(frozen=True)
class InitConfig:
    voxel_size: float = 0.05
    max_background_points: int = 1_000_000
    box_margin: float = 0.25
    dynamic_displacement_threshold: float = 1.0
    dynamic_classes: tuple[str, ...] = ("VEHICLE",)
    min_object_points: int = 2_000
    initial_opacity: float = 0.1


@dataclass(frozen=True)
class FeaturesConfig:
    """Feature flags. Each phase of the roadmap switches one or more of these on."""

    dynamic: bool = False
    pose_refine: bool = False
    sky: bool = False
    exposure: bool = False
    depth_loss: bool = False
    semantics: bool = False
    # Remove pixels of tracked-but-unmodelled actors (pedestrians, cyclists, and every
    # moving vehicle when ``dynamic`` is off) from the photometric loss.
    mask_unmodelled_actors: bool = True


@dataclass(frozen=True)
class ModelConfig:
    sh_degree: int = 3
    sky_resolution: int = 64


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 30_000
    seed: int = 42
    device: str = "cuda"
    strategy: str = "mcmc"  # "mcmc" | "default"
    cap_max: int = 1_500_000
    densify_grad_threshold: float = 0.0005
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    sh_degree_interval: int = 1_000
    lr_means: float = 1.6e-4
    lr_means_final_factor: float = 0.01
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    lr_semantics: float = 5e-3
    lr_pose: float = 1e-4
    lr_exposure: float = 1e-3
    lr_sky: float = 1e-2
    ssim_lambda: float = 0.2
    semantic_weight: float = 0.05
    depth_weight: float = 0.1
    sky_weight: float = 0.05
    opacity_reg_weight: float = 0.01
    scale_reg_weight: float = 0.01
    exposure_reg_weight: float = 1e-3
    pose_smooth_weight: float = 1e-2
    log_every: int = 100
    eval_every: int = 5_000
    checkpoint_every: int = 10_000


@dataclass(frozen=True)
class RenderConfig:
    near_plane: float = 0.1
    far_plane: float = 1_000.0
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rasterize_mode: str = "antialiased"
    packed: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    init: InitConfig = field(default_factory=InitConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output_dir: Path = Path("outputs/default")


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be a mapping")
        return _build(annotation, value, where)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _coerce(value, non_none[0], where)
    if annotation is Path:
        return Path(value)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], where) for v in value)
        if len(args) != len(value):
            raise ValueError(f"{where} must have {len(args)} entries")
        return tuple(_coerce(v, a, where) for v, a in zip(value, args, strict=True))
    if annotation is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if annotation in (int, str, bool) and not isinstance(value, annotation):
        raise ValueError(f"{where} must be {annotation.__name__}, got {value!r}")
    return value


def _build(cls: type, values: dict[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = values.keys() - names
    if unknown:
        raise ValueError(f"unknown field(s) in {where or 'config'}: {', '.join(sorted(unknown))}")
    kwargs = {
        key: _coerce(value, hints[key], f"{where}.{key}" if where else key)
        for key, value in values.items()
    }
    return cls(**kwargs)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must look like section.key=value, got {override!r}")
        path, text = override.split("=", 1)
        node = raw
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _parse_scalar(text)
    return raw


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, raw, "")


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    base = raw.pop("inherit", None)
    if base is not None:
        parent = yaml.safe_load((Path(path).parent / base).read_text()) or {}
        parent.pop("inherit", None)
        raw = _deep_merge(parent, raw)
    return config_from_dict(apply_overrides(raw, overrides or []))


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(dataclasses.asdict(cfg))
