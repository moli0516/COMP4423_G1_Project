"""Integration from native annotations into the existing validation runner manifest."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release(tmp_path):
    annotations = tmp_path / "release" / "annotations" / "pairs.jsonl"
    images = annotations.parent.parent / "photos"
    images.mkdir(parents=True)
    annotations.parent.mkdir()
    for name in ("before", "after"):
        Image.new("RGB", (64, 48)).save(images / f"{name}.png")

    def pair(pid, control=False):
        supported = {"class_id": 0, "bbox_xyxy": [1, 2, 15, 18]}
        has_images = pid.startswith("val")
        return {
            "schema_version": "1.0",
            "pair_id": pid,
            "capture_group_id": pid,
            "width": 64,
            "height": 48,
            "rules_version": "fixture",
            "before_image_id": f"{pid}-before",
            "after_image_id": f"{pid}-after",
            "before_image_path": "../photos/before.png" if has_images else "missing_before",
            "after_image_path": "../photos/after.png" if has_images else "missing_after",
            "independently_captured_no_change": control,
            "manual_before_objects": [{"object_id": "before-bottle", **supported}],
            "manual_after_objects": [{"object_id": "after-bottle", **supported}],
            "correspondences": [
                {
                    "class_id": 0,
                    "before_object_id": "before-bottle",
                    "after_object_id": "after-bottle",
                }
            ],
            "events": [
                {
                    "class_id": 0,
                    "status": "unchanged",
                    "before_object_id": "before-bottle",
                    "after_object_id": "after-bottle",
                }
            ],
        }

    ids = ["train", "val1", "val2", "test"]
    annotations.write_text("".join(json.dumps(pair(i, i == "val2")) + "\n" for i in ids))
    splits = annotations.parent / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 42,
                "dataset_release": "fixture",
                "splits": {"train": ["train"], "val": ["val1", "val2"], "test": ["test"]},
            }
        )
    )
    return annotations, splits, tmp_path / "outputs" / "val.json"


def prepare(files):
    annotations, splits, out = files
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_real_validation.py"),
            "--annotations",
            str(annotations),
            "--splits",
            str(splits),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        cwd=out.parent.parent,
    )


def test_validation_converts_only_val_and_roundtrips_existing_runner(release):
    annotations, _, output = release
    completed = prepare(release)
    assert completed.returncode == 0, completed.stderr
    spec = importlib.util.spec_from_file_location(
        "c_batch", ROOT / "scripts/evaluate_real_pairs.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    loaded = runner.load_manifest(output)
    assert loaded["dataset_release"] == "fixture" and loaded["split"] == "val"
    assert (
        loaded["source_annotations_sha256"] == hashlib.sha256(annotations.read_bytes()).hexdigest()
    )
    assert (
        loaded["source_split_manifest_sha256"]
        == hashlib.sha256(release[1].read_bytes()).hexdigest()
    )
    assert [row["pair_id"] for row in loaded["pairs"]] == ["val1", "val2"]
    images = annotations.parent.parent / "photos"
    for row in loaded["pairs"]:
        assert row["before_path"] == images / "before.png"
        assert row["after_path"] == images / "after.png"
    assert loaded["eligible_no_change_pair_ids"] == ["val2"]
    assert loaded["ground_truth"] == {
        pair_id: [
            {
                "pair_id": pair_id,
                "class_id": 0,
                "status": "unchanged",
                "before_box": [1, 2, 15, 18],
                "after_box": [1, 2, 15, 18],
            }
        ]
        for pair_id in ("val1", "val2")
    }


def test_validation_never_overwrites_existing_manifest(release):
    output = release[2]
    output.parent.mkdir()
    output.write_bytes(b"keep")
    assert prepare(release).returncode != 0
    assert output.read_bytes() == b"keep"


def test_invalid_control_does_not_publish(release):
    annotations, _, output = release
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    for row in rows:
        if row["pair_id"] == "val2":
            row["events"][0]["status"] = "moved"
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    completed = prepare(release)
    assert completed.returncode != 0 and "no-change" in completed.stderr
    assert not output.exists()
