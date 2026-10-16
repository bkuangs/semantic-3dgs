"""Convert Waymo Open Dataset (Perception v1.4.x) .tfrecord segments to the project format.

Runs in a *separate* environment because the official package pins TensorFlow:

    uv venv --python 3.10 .venv-waymo
    uv pip install --python .venv-waymo "waymo-open-dataset-tf-2-12-0==1.6.4" opencv-python-headless
    .venv-waymo/bin/python scripts/preprocess_waymo.py segment-XXXX.tfrecord --out data/scenes

What it does (see ``src/semantic_3dgs/data/scene.py`` for the output schema):
  * re-centres the global frame at the first ego pose (float32 precision);
  * undistorts images with the calibrated Brown-Conrady model, writes a valid-pixel mask;
  * stores the per-image ego pose (rolling-shutter-aware pose at capture), velocity and timing;
  * concatenates first returns of all five LiDARs in the vehicle frame;
  * stores 3D box tracks in each frame's vehicle frame;
  * writes Waymo's own 2D panoptic labels (where present) to ``semantics_gt/``.

Not exercised in CI (no data); validate with ``s3dgs check`` before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_3dgs.classes import build_lookup  # noqa: E402  (pure numpy, no torch needed)


def _mat(transform) -> np.ndarray:
    return np.asarray(transform, dtype=np.float64).reshape(4, 4)


def convert(tfrecord: Path, out_root: Path, cameras: list[str], max_frames: int | None,
            quality: int) -> Path:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2, label_pb2
    from waymo_open_dataset.utils import frame_utils

    try:
        from waymo_open_dataset.protos import camera_segmentation_pb2
        from waymo_open_dataset.utils import camera_segmentation_utils
    except ImportError:  # pragma: no cover - old package versions
        camera_segmentation_pb2 = camera_segmentation_utils = None

    cam_name = dataset_pb2.CameraName.Name.Name
    dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
    out = None
    meta: dict = {"cameras": {}, "frames": []}
    tracks: dict = {}
    undistort: dict = {}
    T_world_global = None
    seg_lut = None
    if camera_segmentation_pb2 is not None:
        names = dict(camera_segmentation_pb2.CameraSegmentation.Type.items())
        seg_lut = build_lookup({v: k.removeprefix("TYPE_") for k, v in names.items()})

    for f, record in enumerate(dataset):
        if max_frames is not None and f >= max_frames:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(record.numpy()))

        if out is None:
            out = out_root / frame.context.name
            out.mkdir(parents=True, exist_ok=True)
            meta["segment"] = frame.context.name
            T_global_world = _mat(frame.pose.transform)
            T_world_global = np.linalg.inv(T_global_world)
            meta["T_global_world"] = T_global_world.tolist()
            for calib in frame.context.camera_calibrations:
                name = cam_name(calib.name)
                if name not in cameras:
                    continue
                fu, fv, cu, cv, k1, k2, p1, p2, k3 = calib.intrinsic
                K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
                dist = np.array([k1, k2, p1, p2, k3])
                size = (calib.width, calib.height)
                maps = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1)
                undistort[name] = maps
                valid = cv2.remap(np.full((calib.height, calib.width), 255, np.uint8), *maps,
                                  cv2.INTER_NEAREST, borderValue=0)
                (out / "masks").mkdir(exist_ok=True)
                cv2.imwrite(str(out / "masks" / f"{name}.png"), valid)
                meta["cameras"][name] = {
                    "K": K.tolist(), "width": calib.width, "height": calib.height,
                    "T_vehicle_camera": _mat(calib.extrinsic.transform).tolist(),
                    "distortion": dist.tolist(),
                    "rolling_shutter_direction": int(calib.rolling_shutter_direction),
                }

        entry = {
            "index": f,
            "timestamp_us": int(frame.timestamp_micros),
            "T_world_vehicle": (T_world_global @ _mat(frame.pose.transform)).tolist(),
            "lidar": f"lidar/{f:04d}.npz",
            "images": {},
        }
        for image in frame.images:
            name = cam_name(image.name)
            if name not in cameras:
                continue
            raw = cv2.imdecode(np.frombuffer(image.image, np.uint8), cv2.IMREAD_COLOR)
            undist = cv2.remap(raw, *undistort[name], cv2.INTER_LINEAR)
            path = out / "images" / name / f"{f:04d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), undist, [cv2.IMWRITE_JPEG_QUALITY, quality])
            entry["images"][name] = {
                "path": str(path.relative_to(out)),
                "timestamp_us": int(round(image.pose_timestamp * 1e6)),
                "T_world_vehicle": (T_world_global @ _mat(image.pose.transform)).tolist(),
                "velocity": [image.velocity.v_x, image.velocity.v_y, image.velocity.v_z,
                             image.velocity.w_x, image.velocity.w_y, image.velocity.w_z],
                "shutter": image.shutter,
                "camera_trigger_time": image.camera_trigger_time,
                "camera_readout_done_time": image.camera_readout_done_time,
            }
            seg = image.camera_segmentation_label
            if seg_lut is not None and seg.panoptic_label:
                panoptic = camera_segmentation_utils.decode_single_panoptic_label_from_proto(seg)
                semantic, _ = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(  # noqa: E501
                    panoptic, seg.panoptic_label_divisor)
                labels = seg_lut[np.clip(semantic[..., 0], 0, len(seg_lut) - 1)]
                labels = cv2.remap(labels, *undistort[name], cv2.INTER_NEAREST, borderValue=255)
                seg_path = out / "semantics_gt" / name / f"{f:04d}.png"
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(seg_path), labels)

        range_images, camera_projections, *_, top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, top_pose, ri_index=0
        )
        lidar = np.concatenate(points).astype(np.float32)
        (out / "lidar").mkdir(exist_ok=True)
        np.savez_compressed(out / "lidar" / f"{f:04d}.npz", points=lidar)

        for label in frame.laser_labels:
            category = label_pb2.Label.Type.Name(label.type).removeprefix("TYPE_")
            track = tracks.setdefault(label.id, {"category": category, "boxes": {}})
            b = label.box
            track["boxes"][str(f)] = {
                "center": [b.center_x, b.center_y, b.center_z],
                "size": [b.length, b.width, b.height],
                "heading": b.heading,
                "speed": [label.metadata.speed_x, label.metadata.speed_y],
                "num_lidar_points": label.num_lidar_points_in_box,
            }
        meta["frames"].append(entry)
        print(f"\r{frame.context.name}: frame {f}", end="", flush=True)

    print()
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "tracks.json").write_text(json.dumps(tracks))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tfrecords", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/scenes"))
    parser.add_argument("--cameras", nargs="+", default=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    for path in args.tfrecords:
        print(f"wrote {convert(path, args.out, args.cameras, args.max_frames, args.jpeg_quality)}")


if __name__ == "__main__":
    main()
