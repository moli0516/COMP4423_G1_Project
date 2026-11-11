"""Integration: annotations -> catalog -> semantic validation ->
evaluator-facing events (review item 10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deskcheck.data.dataset_catalog import load_annotations, load_manifest
from deskcheck.evaluation.annotation_converter import (
    convert_dataset,
    convert_pair,
    ground_truth_events,
)


def _valid_pair(pair_id: str, group_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "pair_id": pair_id,
        "capture_group_id": group_id,
        "before_image_id": f"{pair_id}_b",
        "after_image_id": f"{pair_id}_a",
        "before_image_path": f"raw/{pair_id}_b.jpg",
        "after_image_path": f"raw/{pair_id}_a.jpg",
        "width": 640, "height": 480, "rules_version": "1.0",
        "independently_captured_no_change": False,
        "manual_before_objects": [
            {"object_id": "b_0", "class_id": 0,
             "bbox_xyxy": [50.0, 50.0, 100.0, 100.0]},
            {"object_id": "b_1", "class_id": 1,
             "bbox_xyxy": [200.0, 50.0, 250.0, 100.0]},
            {"object_id": "b_2", "class_id": 2,
             "bbox_xyxy": [100.0, 300.0, 400.0, 400.0]},
        ],
        "manual_after_objects": [
            {"object_id": "a_0", "class_id": 0,
             "bbox_xyxy": [50.0, 50.0, 100.0, 100.0]},
            {"object_id": "a_2", "class_id": 2,
             "bbox_xyxy": [200.0, 300.0, 500.0, 400.0]},
            {"object_id": "a_3", "class_id": 3,
             "bbox_xyxy": [500.0, 100.0, 550.0, 150.0]},
        ],
        "correspondences": [
            {"before_object_id": "b_0", "after_object_id": "a_0", "class_id": 0},
            {"before_object_id": "b_2", "after_object_id": "a_2", "class_id": 2},
        ],
        "events": [
            {"class_id": 0, "status": "unchanged",
             "before_object_id": "b_0", "after_object_id": "a_0"},
            {"class_id": 1, "status": "removed",
             "before_object_id": "b_1", "after_object_id": None},
            {"class_id": 2, "status": "moved",
             "before_object_id": "b_2", "after_object_id": "a_2"},
            {"class_id": 3, "status": "added",
             "before_object_id": None, "after_object_id": "a_3"},
        ],
    }


def _write(tmp_path: Path, records, splits):
    ann = tmp_path / "pairs.jsonl"
    with ann.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    mf = tmp_path / "splits.json"
    mf.write_text(json.dumps({"schema_version": "1.0", "seed": 42,
                              "dataset_release": "v1", "splits": splits}),
                  encoding="utf-8")
    return ann, mf


def test_full_pipeline_jsonl_to_events(tmp_path: Path) -> None:
    ann, mf = _write(
        tmp_path,
        [_valid_pair("p1", "g1"), _valid_pair("p2", "g2")],
        {"train": ["p1"], "val": ["p2"], "test": []},
    )
    pairs, manifest = load_annotations(ann), load_manifest(mf)

    # The audit is tested in test_dataset_validator.py. Here we prove the
    # JSONL -> catalog -> conversion path produces the evaluator format.
    events = ground_truth_events(pairs, list(manifest.splits["val"]))
    assert len(events) == 4
    moved = [e for e in events if e["status"] == "moved"]
    assert len(moved) == 1
    assert moved[0]["before_box"] == [100.0, 300.0, 400.0, 400.0]
    assert moved[0]["after_box"] == [200.0, 300.0, 500.0, 400.0]
    assert moved[0]["class_id"] == 2
    assert moved[0]["pair_id"] == "p2"


def test_added_removed_have_null_boxes(tmp_path: Path) -> None:
    ann, _ = _write(tmp_path, [_valid_pair("p1", "g1")],
                    {"train": ["p1"], "val": [], "test": []})
    pairs = load_annotations(ann)
    events = convert_pair(pairs["p1"])
    by_status = {e["status"]: e for e in events}

    assert by_status["added"]["before_box"] is None
    assert by_status["added"]["after_box"] == [500.0, 100.0, 550.0, 150.0]
    assert by_status["removed"]["before_box"] == [200.0, 50.0, 250.0, 100.0]
    assert by_status["removed"]["after_box"] is None
    assert by_status["moved"]["before_box"] is not None
    assert by_status["moved"]["after_box"] is not None
    assert by_status["unchanged"]["before_box"] is not None
    assert by_status["unchanged"]["after_box"] is not None


def test_empty_pair_produces_empty_events(tmp_path: Path) -> None:
    empty = _valid_pair("p_empty", "g_e")
    empty["manual_before_objects"] = []
    empty["manual_after_objects"] = []
    empty["correspondences"] = []
    empty["events"] = []
    ann, _ = _write(tmp_path, [empty],
                    {"train": ["p_empty"], "val": [], "test": []})
    pairs = load_annotations(ann)
    assert convert_pair(pairs["p_empty"]) == []


def test_convert_dataset_covers_all_pairs(tmp_path: Path) -> None:
    ann, _ = _write(
        tmp_path,
        [_valid_pair("p1", "g1"), _valid_pair("p2", "g2")],
        {"train": ["p1"], "val": ["p2"], "test": []},
    )
    pairs = load_annotations(ann)
    converted = convert_dataset(pairs)
    assert set(converted.keys()) == {"p1", "p2"}
    assert all(len(events) == 4 for events in converted.values())


def test_no_change_pair_with_change_event_rejected(tmp_path: Path) -> None:
    bad = _valid_pair("bad", "gb")
    bad["independently_captured_no_change"] = True
    ann, _ = _write(tmp_path, [bad],
                    {"train": ["bad"], "val": [], "test": []})
    pairs = load_annotations(ann)
    with pytest.raises(ValueError, match="no-change pair contains change events"):
        convert_pair(pairs["bad"])


def test_missing_pair_id_rejected_in_ground_truth(tmp_path: Path) -> None:
    ann, _ = _write(tmp_path, [_valid_pair("p1", "g1")],
                    {"train": ["p1"], "val": [], "test": []})
    pairs = load_annotations(ann)
    with pytest.raises(ValueError, match="not in dataset"):
        ground_truth_events(pairs, ["p_unknown"])
