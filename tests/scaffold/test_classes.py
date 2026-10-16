import numpy as np

from semantic_3dgs.classes import CLASS_INDEX, IGNORE_INDEX, build_lookup, map_label_name


def test_cityscapes_names():
    expected = {
        "road": "road", "sidewalk": "sidewalk", "building": "building", "wall": "building",
        "pole": "traffic_infra", "traffic light": "traffic_infra", "traffic sign": "traffic_infra",
        "vegetation": "vegetation", "terrain": "vegetation", "sky": "sky", "person": "pedestrian",
        "rider": "cyclist", "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
        "motorcycle": "cyclist", "bicycle": "cyclist",
    }
    for name, target in expected.items():
        assert map_label_name(name) == CLASS_INDEX[target], name


def test_mapillary_and_waymo_names():
    assert map_label_name("Lane Marking - General") == CLASS_INDEX["lane_marking"]
    assert map_label_name("Lane Marking - Crosswalk") == CLASS_INDEX["lane_marking"]
    assert map_label_name("Ego Vehicle") == IGNORE_INDEX
    assert map_label_name("Bike Lane") == CLASS_INDEX["road"]
    assert map_label_name("LANE_MARKER") == CLASS_INDEX["lane_marking"]
    assert map_label_name("ROAD_MARKER") == CLASS_INDEX["lane_marking"]
    assert map_label_name("EGO_VEHICLE") == IGNORE_INDEX
    assert map_label_name("Billboard") == CLASS_INDEX["other"]


def test_lookup_table():
    lut = build_lookup({0: "road", 2: "sky"})
    assert lut.dtype == np.uint8
    assert lut.tolist() == [CLASS_INDEX["road"], IGNORE_INDEX, CLASS_INDEX["sky"]]
