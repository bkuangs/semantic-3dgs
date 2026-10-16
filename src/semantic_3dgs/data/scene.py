"""Loader for the preprocessed scene format.

Directory layout written by ``scripts/preprocess_waymo.py`` and ``s3dgs synth``::

    scene_dir/
      meta.json                       cameras, per-frame ego poses, per-image poses
      tracks.json                     3D box tracks (boxes in the *vehicle* frame of each frame)
      images/{CAMERA}/{frame:04d}.jpg undistorted RGB
      masks/{CAMERA}.png              valid-pixel mask after undistortion (optional)
      lidar/{frame:04d}.npz           ``points`` (N, 3) float32 in the vehicle frame
      semantics/{CAMERA}/{frame:04d}.png  uint8 labels in :mod:`semantic_3dgs.classes` (optional)

``meta.json``::

    {
      "segment": str,
      "T_global_world": 4x4,                 # world = vehicle frame at the first frame
      "cameras": {name: {"K": 3x3, "width": int, "height": int,
                         "T_vehicle_camera": 4x4, "distortion": [...]}},
      "frames": [{"index": int, "timestamp_us": int, "T_world_vehicle": 4x4,
                  "lidar": "lidar/0000.npz",
                  "images": {name: {"path": str, "timestamp_us": int,
                                    "T_world_vehicle": 4x4}}}]
    }

``tracks.json``::

    {track_id: {"category": "VEHICLE" | "PEDESTRIAN" | "CYCLIST" | "SIGN",
                "boxes": {frame_index: {"center": [3], "size": [l, w, h], "heading": float}}}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from semantic_3dgs.config import DataConfig


@dataclass(frozen=True)
class CameraCalib:
    name: str
    K: np.ndarray  # (3, 3), already scaled by the configured downscale
    width: int
    height: int
    T_vehicle_camera: np.ndarray  # (4, 4), Waymo camera axes


@dataclass(frozen=True)
class Box:
    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,) length, width, height
    heading: float


@dataclass
class Track:
    track_id: str
    category: str
    boxes: dict[int, Box]  # local frame index -> box in that frame's vehicle frame


@dataclass(frozen=True)
class ImageRecord:
    index: int
    frame: int  # local frame index
    camera: str
    path: Path
    T_world_vehicle: np.ndarray  # (4, 4) ego pose at the image capture time
    timestamp_us: int
    semantics_path: Path | None


def _mat(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


class DrivingScene:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.root = Path(cfg.scene_dir)
        meta = json.loads((self.root / "meta.json").read_text())
        self.segment = meta.get("segment", self.root.name)

        all_frames = meta["frames"]
        start, end = cfg.frame_range if cfg.frame_range is not None else (0, len(all_frames))
        frames = all_frames[start:end]
        if not frames:
            raise ValueError(f"empty frame range {cfg.frame_range} for {self.root}")
        self._source_index = [f["index"] for f in frames]
        local_of_source = {src: i for i, src in enumerate(self._source_index)}

        self.cameras: dict[str, CameraCalib] = {}
        for name in cfg.cameras:
            if name not in meta["cameras"]:
                raise KeyError(f"camera {name} not in {self.root / 'meta.json'}")
            cam = meta["cameras"][name]
            width, height = cam["width"] // cfg.downscale, cam["height"] // cfg.downscale
            scale = np.diag([width / cam["width"], height / cam["height"], 1.0])
            self.cameras[name] = CameraCalib(
                name=name,
                K=scale @ _mat(cam["K"]),
                width=width,
                height=height,
                T_vehicle_camera=_mat(cam["T_vehicle_camera"]),
            )

        self.frame_poses = np.stack([_mat(f["T_world_vehicle"]) for f in frames])
        self.frame_timestamps = np.array([f["timestamp_us"] for f in frames], dtype=np.int64)
        self._lidar_paths = [self.root / f["lidar"] for f in frames]

        self.images: list[ImageRecord] = []
        for local, frame in enumerate(frames):
            for name in cfg.cameras:
                entry = frame["images"].get(name)
                if entry is None:
                    continue
                sem = self.root / cfg.semantics_subdir / name / f"{frame['index']:04d}.png"
                self.images.append(
                    ImageRecord(
                        index=len(self.images),
                        frame=local,
                        camera=name,
                        path=self.root / entry["path"],
                        T_world_vehicle=_mat(entry["T_world_vehicle"]),
                        timestamp_us=int(entry["timestamp_us"]),
                        semantics_path=sem if sem.exists() else None,
                    )
                )

        self.tracks: dict[str, Track] = {}
        tracks_path = self.root / "tracks.json"
        raw_tracks = json.loads(tracks_path.read_text()) if tracks_path.exists() else {}
        for track_id, raw in raw_tracks.items():
            boxes = {}
            for src, box in raw["boxes"].items():
                local = local_of_source.get(int(src))
                if local is None:
                    continue
                boxes[local] = Box(
                    center=_mat(box["center"]), size=_mat(box["size"]), heading=float(box["heading"])
                )
            if boxes:
                self.tracks[track_id] = Track(track_id, raw["category"], boxes)

        self._image_cache: dict[int, Tensor] = {}
        self._valid_masks: dict[str, Tensor] = {}

    # ------------------------------------------------------------------ sizes
    @property
    def num_frames(self) -> int:
        return len(self.frame_poses)

    @property
    def num_images(self) -> int:
        return len(self.images)

    def source_frame_index(self, local: int) -> int:
        return self._source_index[local]

    def scene_extent(self) -> float:
        """Radius of the ego trajectory (+ a floor), used to scale learning rates/densification."""
        positions = self.frame_poses[:, :3, 3]
        radius = np.linalg.norm(positions - positions.mean(0), axis=1).max()
        return float(max(radius * 1.1, 10.0))

    # ------------------------------------------------------------------ loaders
    def image(self, index: int) -> Tensor:
        """(H, W, 3) float32 in [0, 1] on CPU."""
        if index in self._image_cache:
            return self._image_cache[index].float() / 255.0
        rec = self.images[index]
        cam = self.cameras[rec.camera]
        img = Image.open(rec.path).convert("RGB")
        if img.size != (cam.width, cam.height):
            img = img.resize((cam.width, cam.height), Image.BILINEAR)
        tensor = torch.from_numpy(np.asarray(img).copy())
        if self.cfg.cache_images:
            self._image_cache[index] = tensor
        return tensor.float() / 255.0

    def semantics(self, index: int) -> Tensor | None:
        """(H, W) int64 labels, or ``None`` if no semantic map exists for this image."""
        rec = self.images[index]
        if rec.semantics_path is None:
            return None
        cam = self.cameras[rec.camera]
        img = Image.open(rec.semantics_path)
        if img.size != (cam.width, cam.height):
            img = img.resize((cam.width, cam.height), Image.NEAREST)
        return torch.from_numpy(np.asarray(img).astype(np.int64))

    def valid_mask(self, camera: str) -> Tensor:
        """(H, W) bool mask of pixels that are valid after undistortion."""
        if camera not in self._valid_masks:
            cam = self.cameras[camera]
            path = self.root / "masks" / f"{camera}.png"
            if path.exists():
                img = Image.open(path).convert("L").resize((cam.width, cam.height), Image.NEAREST)
                mask = torch.from_numpy(np.asarray(img) > 127)
            else:
                mask = torch.ones(cam.height, cam.width, dtype=torch.bool)
            self._valid_masks[camera] = mask
        return self._valid_masks[camera]

    def lidar(self, frame: int) -> Tensor:
        """(N, 3) float32 LiDAR points in the vehicle frame of ``frame``."""
        with np.load(self._lidar_paths[frame]) as data:
            return torch.from_numpy(data["points"].astype(np.float32))

    # ------------------------------------------------------------------ tensors
    def K(self, camera: str) -> Tensor:
        return torch.from_numpy(self.cameras[camera].K).float()

    def T_vehicle_camera(self, camera: str) -> Tensor:
        return torch.from_numpy(self.cameras[camera].T_vehicle_camera).float()

    def T_world_vehicle(self, frame: int) -> Tensor:
        """Ego pose at the LiDAR timestamp of ``frame`` (boxes are expressed in this frame)."""
        return torch.from_numpy(self.frame_poses[frame]).float()

    def image_T_world_vehicle(self, index: int) -> Tensor:
        """Ego pose at the capture time of image ``index``."""
        return torch.from_numpy(self.images[index].T_world_vehicle).float()
