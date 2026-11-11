"""Independently verify the persisted synthetic demo before publishing it."""

import json
from pathlib import Path

from PIL import Image

from deskcheck.application.app_config import AppConfig
from deskcheck.application.result_serializer import ResultSerializer

EXPECTED_STATUSES = {0: "moved", 1: "removed", 2: "unchanged", 3: "added"}
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
    statuses = {record["class_id"]: record["status"] for record in records}
    changes = sum(record["indicates_change"] for record in records)
    if len(records) != 4 or statuses != EXPECTED_STATUSES or changes != 3:
        raise ValueError("Demo output does not match independently expected fixture statuses")

    if (bundle / "result.json").read_bytes() != serializer.to_json(validated).encode("utf-8"):
        raise ValueError("Demo JSON bytes differ from canonical serialization")
    if (bundle / "result.csv").read_bytes() != serializer.to_csv(validated).encode("utf-8"):
        raise ValueError("Demo CSV bytes differ from canonical serialization")

    for name in ("annotated_before.png", "annotated_after.png"):
        with Image.open(bundle / name) as img:
            if img.format != "PNG" or img.size != (160, 120):
                raise ValueError(f"Invalid demo annotation dimensions or format: {name}")
            img.verify()
    return len(records), changes
