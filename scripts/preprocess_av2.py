"""Convert Argoverse 2 Sensor Dataset logs to the project format.

The AV2 sensor data is on a public S3 bucket (no account needed). This script can download just the
files it needs for a window of frames, then convert them:

    uv run --with pyarrow --with opencv-python-headless python scripts/preprocess_av2.py \\
        LOG_ID --split val --download --max-frames 60 --out data/scenes/av2

``LOG_ID`` may also be a path to an already-downloaded log directory.

What it does (see ``src/semantic_3dgs/data/scene.py`` for the output schema):
  * one frame per LiDAR sweep (10 Hz); each camera contributes its image closest in time (20 Hz);
  * re-centres the city frame at the first ego pose (float32 precision);
  * converts AV2's OpenCV-axis camera extrinsics to Waymo camera axes (x fwd, y left, z up);
  * undistorts images with the radial (k1, k2, k3) model and writes a valid-pixel mask (which also
    excludes the ego hood visible at the bottom of the front camera);
  * stores each image's own ego pose at its capture time;
  * stores 3D cuboid tracks in each sweep's vehicle frame, with categories mapped to the
    project's VEHICLE / PEDESTRIAN / CYCLIST / SIGN / OTHER.

AV2 ring cameras are renamed to the Waymo-style names used by the configs (``FRONT``,
``FRONT_LEFT``, ...). Validate the output with ``s3dgs check`` before training.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BUCKET_URL = "https://s3.amazonaws.com/argoverse"
SENSOR_PREFIX = "datasets/av2/sensor"

CAMERAS = {
    "FRONT": "ring_front_center",
    "FRONT_LEFT": "ring_front_left",
    "FRONT_RIGHT": "ring_front_right",
    "SIDE_LEFT": "ring_side_left",
    "SIDE_RIGHT": "ring_side_right",
    "REAR_LEFT": "ring_rear_left",
    "REAR_RIGHT": "ring_rear_right",
}

CATEGORY_MAP = {
    "VEHICLE": [
        "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "SCHOOL_BUS", "ARTICULATED_BUS", "BOX_TRUCK",
        "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER", "RAILED_VEHICLE",
    ],
    "PEDESTRIAN": ["PEDESTRIAN", "OFFICIAL_SIGNALER", "STROLLER", "WHEELCHAIR", "DOG", "ANIMAL"],
    "CYCLIST": ["BICYCLE", "BICYCLIST", "MOTORCYCLE", "MOTORCYCLIST", "WHEELED_RIDER", "WHEELED_DEVICE"],
    "SIGN": ["SIGN", "STOP_SIGN", "MOBILE_PEDESTRIAN_CROSSING_SIGN"],
}
CATEGORY_OF = {av2: ours for ours, names in CATEGORY_MAP.items() for av2 in names}

# OpenCV camera axes (x right, y down, z forward) -> Waymo camera axes (x forward, y left, z up).
T_WAYMO_CAM_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

# Images further than this from their sweep are dropped (cameras run at 20 Hz -> <= 25 ms).
MAX_IMAGE_OFFSET_NS = 60_000_000

# The ego vehicle's hood covers the bottom ~8% of ring_front_center. It is fixed to the camera,
# not the world, so it is excluded from the valid-pixel mask (fraction of image height).
EGO_HOOD_FRACTION = {"FRONT": 0.10}


# ---------------------------------------------------------------------------------- download
def _list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET_URL}?{urllib.parse.urlencode(query)}", timeout=120) as r:
            text = r.read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", text)
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not match:
            return keys
        token = match.group(1)


def _fetch(key: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BUCKET_URL}/{key}", timeout=300) as r, tmp.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)


def _stamp(key: str) -> int:
    return int(Path(key).stem)


def download_log(log_id: str, split: str, raw_root: Path, cameras: list[str], start: int,
                 max_frames: int | None, workers: int) -> tuple[Path, list[int]]:
    """Fetch metadata, the selected LiDAR sweeps, and the closest image per camera and sweep.

    Returns the log directory and the timestamps of the selected sweeps.
    """
    prefix = f"{SENSOR_PREFIX}/{split}/{log_id}"
    log_dir = raw_root / split / log_id
    jobs = [
        (f"{prefix}/{name}", log_dir / name)
        for name in ("annotations.feather", "city_SE3_egovehicle.feather",
                     "calibration/egovehicle_SE3_sensor.feather", "calibration/intrinsics.feather")
    ]
    sweeps = sorted(_list_keys(f"{prefix}/sensors/lidar/"), key=_stamp)
    if not sweeps:
        raise FileNotFoundError(f"no LiDAR sweeps under s3://argoverse/{prefix} (wrong split or log id?)")
    end = None if max_frames is None else start + max_frames
    sweeps = sweeps[start:end]
    jobs += [(k, log_dir / "sensors" / "lidar" / Path(k).name) for k in sweeps]
    for cam in cameras:
        av2_name = CAMERAS[cam]
        keys = _list_keys(f"{prefix}/sensors/cameras/{av2_name}/")
        stamps = np.array([_stamp(k) for k in keys], dtype=np.int64)
        for sweep in sweeps:
            k = keys[int(np.abs(stamps - _stamp(sweep)).argmin())]
            jobs.append((k, log_dir / "sensors" / "cameras" / av2_name / Path(k).name))
    jobs = [(k, d) for k, d in jobs if not d.exists()]
    print(f"downloading {len(jobs)} files for {log_id} ({len(sweeps)} sweeps, {len(cameras)} cameras)")
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(lambda job: _fetch(*job), jobs), 1):
            print(f"\r  {i}/{len(jobs)}", end="", flush=True)
    print()
    return log_dir, [_stamp(k) for k in sweeps]


# ---------------------------------------------------------------------------------- geometry
def _quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    q = np.stack([qw, qx, qy, qz], axis=-1).astype(np.float64)
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)


def _se3(table: dict, i=slice(None)) -> np.ndarray:
    """4x4 transforms from AV2 (qw, qx, qy, qz, tx_m, ty_m, tz_m) columns."""
    R = _quat_to_rotmat(*(np.asarray(table[c])[i] for c in ("qw", "qx", "qy", "qz")))
    t = np.stack([np.asarray(table[c])[i] for c in ("tx_m", "ty_m", "tz_m")], axis=-1)
    T = np.zeros((*R.shape[:-2], 4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def _yaw(qw, qx, qy, qz) -> np.ndarray:
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class _PoseTable:
    """City-from-ego poses, interpolated (lerp + slerp) at arbitrary timestamps."""

    def __init__(self, table: dict):
        order = np.argsort(table["timestamp_ns"])
        self.stamps = np.asarray(table["timestamp_ns"], dtype=np.int64)[order]
        self.quats = np.stack([np.asarray(table[c])[order] for c in ("qw", "qx", "qy", "qz")], -1)
        self.trans = np.stack([np.asarray(table[c])[order] for c in ("tx_m", "ty_m", "tz_m")], -1)

    def __call__(self, stamp: int) -> np.ndarray:
        j = int(np.clip(np.searchsorted(self.stamps, stamp), 1, len(self.stamps) - 1))
        t0, t1 = self.stamps[j - 1], self.stamps[j]
        a = float(np.clip((stamp - t0) / max(t1 - t0, 1), 0.0, 1.0))
        q0, q1 = self.quats[j - 1], self.quats[j]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        theta = np.arccos(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if theta < 1e-6:
            q = (1 - a) * q0 + a * q1
        else:
            q = (np.sin((1 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)
        T = np.eye(4)
        T[:3, :3] = _quat_to_rotmat(*q)
        T[:3, 3] = (1 - a) * self.trans[j - 1] + a * self.trans[j]
        return T


# ---------------------------------------------------------------------------------- convert
def convert(log_dir: Path, out_root: Path, cameras: list[str], start: int, max_frames: int | None,
            quality: int, sweep_stamps: list[int] | None = None) -> Path:
    """Convert a log directory. ``sweep_stamps`` (if given) selects sweeps explicitly instead of
    ``start`` / ``max_frames``."""
    import cv2
    import pyarrow.feather as feather

    def read(path: Path) -> dict:
        return feather.read_table(path).to_pydict()

    log_id = log_dir.name
    out = out_root / log_id
    out.mkdir(parents=True, exist_ok=True)

    poses = _PoseTable(read(log_dir / "city_SE3_egovehicle.feather"))
    extrinsics = read(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    intrinsics = read(log_dir / "calibration" / "intrinsics.feather")

    sweeps = sorted((log_dir / "sensors" / "lidar").glob("*.feather"), key=lambda p: int(p.stem))
    if sweep_stamps is not None:
        wanted = set(sweep_stamps)
        sweeps = [p for p in sweeps if int(p.stem) in wanted]
    else:
        sweeps = sweeps[start:None if max_frames is None else start + max_frames]
    if not sweeps:
        raise FileNotFoundError(f"no LiDAR sweeps in {log_dir / 'sensors' / 'lidar'}")
    stamps = [int(p.stem) for p in sweeps]

    T_global_world = poses(stamps[0])
    T_world_global = np.linalg.inv(T_global_world)
    meta: dict = {"segment": log_id, "T_global_world": T_global_world.tolist(), "cameras": {},
                  "frames": [], "source": "argoverse2"}

    undistort, image_stamps = {}, {}
    for cam in cameras:
        av2_name = CAMERAS[cam]
        i = intrinsics["sensor_name"].index(av2_name)
        width, height = int(intrinsics["width_px"][i]), int(intrinsics["height_px"][i])
        K = np.array([[intrinsics["fx_px"][i], 0, intrinsics["cx_px"][i]],
                      [0, intrinsics["fy_px"][i], intrinsics["cy_px"][i]], [0, 0, 1]])
        dist = np.array([intrinsics["k1"][i], intrinsics["k2"][i], 0.0, 0.0, intrinsics["k3"][i]])
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_32FC1)
        undistort[cam] = maps
        valid = cv2.remap(np.full((height, width), 255, np.uint8), *maps, cv2.INTER_NEAREST, borderValue=0)
        if cam in EGO_HOOD_FRACTION:
            valid[int(round(height * (1.0 - EGO_HOOD_FRACTION[cam]))):] = 0
        (out / "masks").mkdir(exist_ok=True)
        cv2.imwrite(str(out / "masks" / f"{cam}.png"), valid)

        j = extrinsics["sensor_name"].index(av2_name)
        T_vehicle_cam_cv = _se3(extrinsics, j)
        meta["cameras"][cam] = {
            "K": K.tolist(), "width": width, "height": height,
            "T_vehicle_camera": (T_vehicle_cam_cv @ np.linalg.inv(T_WAYMO_CAM_FROM_OPENCV)).tolist(),
            "distortion": dist.tolist(),
            "av2_name": av2_name,
        }
        files = sorted((log_dir / "sensors" / "cameras" / av2_name).glob("*.jpg"), key=lambda p: int(p.stem))
        image_stamps[cam] = (np.array([int(p.stem) for p in files], dtype=np.int64), files)

    ann = read(log_dir / "annotations.feather")
    ann_stamps = np.asarray(ann["timestamp_ns"], dtype=np.int64)
    yaws = _yaw(*(np.asarray(ann[c]) for c in ("qw", "qx", "qy", "qz")))
    tracks: dict = {}

    for f, (sweep, stamp) in enumerate(zip(sweeps, stamps, strict=True)):
        entry = {
            "index": f,
            "timestamp_us": stamp // 1000,
            "T_world_vehicle": (T_world_global @ poses(stamp)).tolist(),
            "lidar": f"lidar/{f:04d}.npz",
            "images": {},
        }
        for cam in cameras:
            cam_stamps, files = image_stamps[cam]
            if len(files) == 0:
                continue
            k = int(np.abs(cam_stamps - stamp).argmin())
            if abs(int(cam_stamps[k]) - stamp) > MAX_IMAGE_OFFSET_NS:
                continue
            raw = cv2.imread(str(files[k]), cv2.IMREAD_COLOR)
            undist = cv2.remap(raw, *undistort[cam], cv2.INTER_LINEAR)
            path = out / "images" / cam / f"{f:04d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), undist, [cv2.IMWRITE_JPEG_QUALITY, quality])
            entry["images"][cam] = {
                "path": str(path.relative_to(out)),
                "timestamp_us": int(cam_stamps[k]) // 1000,
                "T_world_vehicle": (T_world_global @ poses(int(cam_stamps[k]))).tolist(),
            }

        lidar = feather.read_table(sweep, columns=["x", "y", "z"]).to_pydict()
        points = np.stack([lidar["x"], lidar["y"], lidar["z"]], axis=-1).astype(np.float32)
        (out / "lidar").mkdir(exist_ok=True)
        np.savez_compressed(out / "lidar" / f"{f:04d}.npz", points=points)

        for r in np.flatnonzero(ann_stamps == stamp):
            av2_category = ann["category"][r]
            track = tracks.setdefault(ann["track_uuid"][r], {
                "category": CATEGORY_OF.get(av2_category, "OTHER"), "av2_category": av2_category,
                "boxes": {},
            })
            track["boxes"][str(f)] = {
                "center": [float(ann["tx_m"][r]), float(ann["ty_m"][r]), float(ann["tz_m"][r])],
                "size": [float(ann["length_m"][r]), float(ann["width_m"][r]), float(ann["height_m"][r])],
                "heading": float(yaws[r]),
                "num_lidar_points": int(ann["num_interior_pts"][r]),
            }
        meta["frames"].append(entry)
        print(f"\r{log_id}: frame {f + 1}/{len(sweeps)}", end="", flush=True)

    print()
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "tracks.json").write_text(json.dumps(tracks))
    _summarize(meta, tracks)
    return out


def _summarize(meta: dict, tracks: dict) -> None:
    """Print how many tracks move in the world frame - the candidates for dynamic nodes."""
    poses = {f["index"]: np.asarray(f["T_world_vehicle"]) for f in meta["frames"]}
    moving: dict[str, int] = {}
    for track in tracks.values():
        centers = np.array([poses[int(f)][:3, :3] @ b["center"] + poses[int(f)][:3, 3]
                            for f, b in track["boxes"].items()])
        if len(centers) > 1 and np.linalg.norm(centers[:, None] - centers[None], axis=-1).max() > 1.0:
            moving[track["category"]] = moving.get(track["category"], 0) + 1
    images = sum(len(f["images"]) for f in meta["frames"])
    print(f"{len(meta['frames'])} frames, {images} images, {len(tracks)} tracks; "
          f"moving (>1 m): {moving or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logs", nargs="+", help="AV2 log ids (with --download) or local log directories")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--download", action="store_true", help="fetch missing files from the public bucket")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/av2"), help="download location")
    parser.add_argument("--out", type=Path, default=Path("data/scenes/av2"))
    parser.add_argument("--cameras", nargs="+", default=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"],
                        choices=list(CAMERAS))
    parser.add_argument("--start-frame", type=int, default=0, help="first LiDAR sweep to use")
    parser.add_argument("--max-frames", type=int, help="number of LiDAR sweeps (10 Hz) to use")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads")
    args = parser.parse_args()
    for log in args.logs:
        log_dir, stamps = Path(log), None
        if args.download:
            log_dir, stamps = download_log(log_dir.name, args.split, args.raw, args.cameras,
                                           args.start_frame, args.max_frames, args.workers)
        out = convert(log_dir, args.out, args.cameras, args.start_frame, args.max_frames,
                      args.jpeg_quality, sweep_stamps=stamps)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
