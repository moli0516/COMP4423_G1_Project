"""Independently verify the persisted synthetic demo before publishing it."""

import json
from math import isclose
from pathlib import Path

from PIL import Image

from deskcheck.application.app_config import AppConfig
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.result_renderer import ResultRenderer
from deskcheck.application.result_serializer import ResultSerializer

# Independent fixture coordinates, not inferred from detector output or filenames.
# Each entry is (status, before_box, after_box, distance_norm, indicates_change).
EXPECTED_EVENTS = {
    0: ("moved", [10, 10, 25, 25], [80, 10, 95, 25], 0.35, True),
    1: ("removed", [10, 40, 25, 55], None, None, True),
    2: ("unchanged", [10, 70, 25, 85], [10, 70, 25, 85], 0.0, False),
    3: ("added", None, [80, 40, 95, 55], None, True),
}
EXPECTED_BUNDLE = {
    "result.json",
    "result.csv",
    "annotated_before.png",
    "annotated_after.png",
}


def verify_demo(staging: Path, expected_before: bytes, expected_after: bytes, config_path: Path) -> tuple[int, int]:
    """Return verified counts; reject incomplete, misleading or altered demo output."""
    for name, original in (("before.png", expected_before), ("after.png", expected_after)):
        path = staging / name
        if not path.is_file() or path.is_symlink() or path.read_bytes() != original:
            raise ValueError(f"Demo source was changed or is missing: {name}")

    bundle = staging / "bundle"
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("Demo CLI did not publish an output bundle")
    files = {p.name for p in bundle.iterdir()}
    if files != EXPECTED_BUNDLE or any(
        not (bundle / name).is_file() or (bundle / name).is_symlink() for name in files
    ):
        raise ValueError("Demo output bundle is incomplete or contains unexpected files")

    result = json.loads((bundle / "result.json").read_bytes().decode("utf-8"))
    serializer = ResultSerializer()
    validated = serializer.snapshot(result)
    config = AppConfig.load(config_path)
    meta = validated.get("metadata")
    if (
        validated["pair_id"] != "c-demo"
        or validated["image_size"] != [160, 120]
        or not isinstance(meta, dict)
        or meta.get("mode") != "fake"
        or meta.get("checkpoint_sha256") is not None
        or meta.get("config_sha256") != config.config_sha256
        or meta.get("rules_version") != config.rules_version
        or meta.get("confidence_threshold") != config.confidence_threshold
        or meta.get("movement_threshold") != config.movement_threshold
    ):
        raise ValueError("Demo output configuration or provenance does not match the fixture")

    records = validated["records"]
    changes = sum(record["indicates_change"] for record in records)
    if len(records) != 4 or {r["class_id"] for r in records} != set(EXPECTED_EVENTS):
        raise ValueError("Demo output does not match independently expected fixture classes")
    for record in records:
        class_id = record["class_id"]
        status, before_box, after_box, distance, indicates_change = EXPECTED_EVENTS[class_id]
        observed_distance = record["distance_norm"]
        distance_matches = (
            observed_distance is None if distance is None
            else type(observed_distance) in (int, float)
            and isclose(observed_distance, distance, rel_tol=0, abs_tol=1e-12)
        )
        if (
            record["status"] != status
            or record["before_box"] != before_box
            or record["after_box"] != after_box
            or not distance_matches
            or record["indicates_change"] is not indicates_change
            or record["movement_threshold"] != config.movement_threshold
            or record["before_detection_id"] != (f"fake-{class_id}" if before_box else None)
            or record["after_detection_id"] != (f"fake-{class_id}" if after_box else None)
        ):
            raise ValueError(f"Demo fixture geometry or object evidence differs for class {class_id}")
    if changes != 3:
        raise ValueError("Demo fixture change count differs from expected")

    if (bundle / "result.json").read_bytes() != serializer.to_json(validated).encode("utf-8"):
        raise ValueError("Demo JSON bytes differ from canonical serialization")
    if (bundle / "result.csv").read_bytes() != serializer.to_csv(validated).encode("utf-8"):
        raise ValueError("Demo CSV bytes differ from canonical serialization")

    frames = ImagePairLoader().load(expected_before, expected_after, "c-demo")
    expected_images = ResultRenderer().render(*frames, validated)
    for name, expected_image in zip(
        ("annotated_before.png", "annotated_after.png"), expected_images
    ):
        with Image.open(bundle / name) as img:
            if img.format != "PNG" or img.size != (160, 120):
                raise ValueError(f"Invalid demo annotation dimensions or format: {name}")
            img.load()
            if img.mode != expected_image.mode or img.tobytes() != expected_image.tobytes():
                raise ValueError(f"Demo annotation pixels differ from rendered fixture: {name}")
    return len(records), changes
