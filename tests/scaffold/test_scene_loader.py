import numpy as np
import torch
from conftest import FOCAL, HEIGHT, NUM_FRAMES, WIDTH

from semantic_3dgs.config import DataConfig
from semantic_3dgs.data.scene import DrivingScene
from semantic_3dgs.data.splits import split_images


def test_loader_reads_everything(driving_scene):
    s = driving_scene
    assert s.num_frames == NUM_FRAMES and s.num_images == NUM_FRAMES
    assert s.image(0).shape == (HEIGHT, WIDTH, 3)
    assert 0.0 <= float(s.image(0).min()) and float(s.image(0).max()) <= 1.0
    assert s.semantics(0).shape == (HEIGHT, WIDTH) and s.semantics(1) is None
    assert s.lidar(0).shape[1] == 3
    assert set(s.tracks) == {"mover", "parked", "walker"}
    assert s.tracks["mover"].category == "VEHICLE"
    assert s.valid_mask("FRONT").all()


def test_downscale_scales_intrinsics(fake_scene):
    s = DrivingScene(DataConfig(scene_dir=fake_scene.root, cameras=("FRONT",), downscale=2))
    cam = s.cameras["FRONT"]
    assert (cam.width, cam.height) == (WIDTH // 2, HEIGHT // 2)
    np.testing.assert_allclose(cam.K[0, 0], FOCAL / 2)
    assert s.image(0).shape == (HEIGHT // 2, WIDTH // 2, 3)


def test_frame_range_reindexes_tracks(fake_scene):
    s = DrivingScene(DataConfig(scene_dir=fake_scene.root, cameras=("FRONT",), frame_range=(2, 4)))
    assert s.num_frames == 2
    assert sorted(s.tracks["mover"].boxes) == [0, 1]
    assert s.source_frame_index(0) == 2
    torch.testing.assert_close(s.T_world_vehicle(0)[0, 3], torch.tensor(2.0))


def test_splits(driving_scene):
    train, test = split_images(driving_scene, holdout_every=2)
    assert sorted(train + test) == list(range(driving_scene.num_images))
    assert not set(train) & set(test)
    assert [driving_scene.images[i].frame for i in test] == [1, 3]
