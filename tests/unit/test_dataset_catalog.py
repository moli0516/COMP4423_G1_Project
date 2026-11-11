"""Unit tests for the annotation/manifest catalog loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deskcheck.data.dataset_catalog import (
    SCHEMA_VERSION,
    load_annotations,
    load_catalog,
    load_manifest,
)


def _base_pair_dict(pair_id: str, group_id: str = "g1") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "capture_group_id": group_id,
        "before_image_id": f"{pair_id}_b",
        "after_image_id": f"{pair_id}_a",
        "before_image_path": f"raw/{pair_id}_b.jpg",
        "after_image_path": f"raw/{pair_id}_a.jpg",
        "width": 640,
        "height": 480,
        "rules_version": "1.0",
        "independently_captured_no_change": False,
        "manual_before_objects": [
            {"object_id": "o1", "class_id": 0, "bbox_xyxy": [10, 10, 50, 50]}
        ],
        "manual_after_objects": [
            {"object_id": "o2", "class_id": 0, "bbox_xyxy": [20, 20, 60, 60]}
        ],
        "correspondences": [
            {"before_object_id": "o1", "after_object_id": "o2", "class_id": 0}
        ],
        "events": [
            {
                "class_id": 0,
                "status": "moved",
                "before_object_id": "o1",
                "after_object_id": "o2",
            }
        ],
    }


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def test_load_annotations_ok(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "pairs.jsonl",
        [_base_pair_dict("p1"), _base_pair_dict("p2", group_id="g2")],
    )
    pairs = load_annotations(path)
    assert set(pairs) == {"p1", "p2"}
    assert pairs["p1"].width == 640
    assert pairs["p1"].events[0]["status"] == "moved"


def test_load_annotations_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(_base_pair_dict("p1")) + "\n\n" + json.dumps(_base_pair_dict("p2")) + "\n",
        encoding="utf-8",
    )
    pairs = load_annotations(path)
    assert set(pairs) == {"p1", "p2"}


def test_load_annotations_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_annotations(tmp_path / "missing.jsonl")


def test_load_annotations_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_annotations(path)


def test_load_annotations_duplicate_pair_id_raises(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "pairs.jsonl",
        [_base_pair_dict("p1"), _base_pair_dict("p1")],
    )
    with pytest.raises(ValueError, match="duplicate pair_id"):
        load_annotations(path)


def test_load_annotations_missing_required_field_raises(tmp_path: Path) -> None:
    bad = _base_pair_dict("p1")
    del bad["events"]
    path = _write_jsonl(tmp_path / "pairs.jsonl", [bad])
    with pytest.raises(ValueError, match="missing fields"):
        load_annotations(path)


def test_load_annotations_invalid_status_raises(tmp_path: Path) -> None:
    bad = _base_pair_dict("p1")
    bad["events"][0]["status"] = "teleported"
    path = _write_jsonl(tmp_path / "pairs.jsonl", [bad])
    with pytest.raises(ValueError, match="invalid status"):
        load_annotations(path)


def test_load_annotations_wrong_schema_version_raises(tmp_path: Path) -> None:
    bad = _base_pair_dict("p1")
    bad["schema_version"] = "0.9"
    path = _write_jsonl(tmp_path / "pairs.jsonl", [bad])
    with pytest.raises(ValueError, match="schema_version"):
        load_annotations(path)


def test_load_manifest_ok(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 42,
                "dataset_release": "v1",
                "splits": {"train": ["p1"], "val": ["p2"], "test": ["p3"]},
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    assert manifest.seed == 42
    assert manifest.splits["val"] == ("p2",)


def test_load_manifest_missing_split_raises(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 42,
                "dataset_release": "v1",
                "splits": {"train": ["p1"], "val": ["p2"]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train, val, test"):
        load_manifest(path)


def test_load_manifest_duplicate_pair_across_splits_raises(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 42,
                "dataset_release": "v1",
                "splits": {"train": ["p1"], "val": ["p1"], "test": []},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="appears in both"):
        load_manifest(path)


def test_load_catalog_cross_check_missing_pair_raises(tmp_path: Path) -> None:
    pairs_path = _write_jsonl(tmp_path / "pairs.jsonl", [_base_pair_dict("p1")])
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 42,
                "dataset_release": "v1",
                "splits": {"train": ["p1"], "val": ["p_unknown"], "test": []},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown pair_ids"):
        load_catalog(pairs_path, manifest_path)


def test_load_catalog_ok(tmp_path: Path) -> None:
    pairs_path = _write_jsonl(
        tmp_path / "pairs.jsonl",
        [_base_pair_dict("p1"), _base_pair_dict("p2", group_id="g2")],
    )
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 42,
                "dataset_release": "v1",
                "splits": {"train": ["p1"], "val": ["p2"], "test": []},
            }
        ),
        encoding="utf-8",
    )
    pairs, manifest = load_catalog(pairs_path, manifest_path)
    assert set(pairs) == {"p1", "p2"}
    assert manifest.splits["train"] == ("p1",)

def test_annotations_reject_string_booleans(tmp_path: Path) -> None:
    rec = _base_pair_dict("p1")
    rec["independently_captured_no_change"] = "false"
    path = _write_jsonl(tmp_path / "pairs.jsonl", [rec])
    with pytest.raises(ValueError, match="must be an actual boolean"):
        load_annotations(path)
