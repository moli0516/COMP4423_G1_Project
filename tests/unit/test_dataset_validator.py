"""Unit tests for dataset_validator.audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deskcheck.data.dataset_catalog import load_annotations, load_manifest
from deskcheck.data.dataset_validator import audit, REQUIRED_CLASS_IDS


def _pair(
    pair_id: str,
    group_id: str,
    *,
    before_classes: list[int],
    after_classes: list[int],
    statuses: list[str],
    no_change: bool = False,
    before_image_id: str | None = None,
    after_image_id: str | None = None,
) -> dict:
    before_objects = [
        {"object_id": f"b{cls}", "class_id": cls,
         "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]}
        for cls in before_classes
    ]
    after_objects = [
        {"object_id": f"a{cls}", "class_id": cls,
         "bbox_xyxy": [20.0, 20.0, 60.0, 60.0]}
        for cls in after_classes
    ]
    events = []
    for cls, status in zip(sorted(set(before_classes) | set(after_classes)), statuses):
        events.append({
            "class_id": cls,
            "status": status,
            "before_object_id": f"b{cls}" if cls in before_classes else None,
            "after_object_id": f"a{cls}" if cls in after_classes else None,
        })
    return {
        "schema_version": "1.0",
        "pair_id": pair_id,
        "capture_group_id": group_id,
        "before_image_id": before_image_id or f"{pair_id}_b",
        "after_image_id": after_image_id or f"{pair_id}_a",
        "before_image_path": f"raw/{pair_id}_b.jpg",
        "after_image_path": f"raw/{pair_id}_a.jpg",
        "width": 640,
        "height": 480,
        "rules_version": "1.0",
        "independently_captured_no_change": no_change,
        "manual_before_objects": before_objects,
        "manual_after_objects": after_objects,
        "correspondences": [],
        "events": events,
    }


def _write(tmp_path: Path, records: list[dict], splits: dict[str, list[str]]) -> tuple:
    pairs_path = tmp_path / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": "1.0", "seed": 42, "dataset_release": "v1",
            "splits": splits,
        }),
        encoding="utf-8",
    )
    return load_annotations(pairs_path), load_manifest(manifest_path)


def _minimal_healthy_dataset(tmp_path: Path) -> tuple:
    """One pair per split, every split has all 6 classes and val/test have all statuses."""
    records = []
    all_classes = list(REQUIRED_CLASS_IDS)
    statuses = ["added", "removed", "moved", "unchanged", "moved", "moved"]
    for split_name, pair_ids in (
        ("train", ["p1", "p2"]),
        ("val", ["p3", "p4"]),
        ("test", ["p5", "p6"]),
    ):
        for pid in pair_ids:
            records.append(_pair(
                pid, f"g_{pid}",
                before_classes=all_classes,
                after_classes=all_classes,
                statuses=statuses,
                no_change=(split_name in ("val", "test") and pid in (pair_ids[1],)),
            ))
    splits = {"train": ["p1", "p2"], "val": ["p3", "p4"], "test": ["p5", "p6"]}
    return _write(tmp_path, records, splits)


def test_audit_healthy_dataset_passes(tmp_path: Path) -> None:
    pairs, manifest = _minimal_healthy_dataset(tmp_path)
    report = audit(pairs, manifest)
    assert report.ok, report.errors
    assert report.pair_counts == {"train": 2, "val": 2, "test": 2}
    assert report.independent_no_change_counts["val"] >= 1
    assert report.independent_no_change_counts["test"] >= 1


def test_audit_empty_dataset_fails(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("", encoding="utf-8")
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0", "seed": 42, "dataset_release": "v1",
                    "splits": {"train": [], "val": [], "test": []}}),
        encoding="utf-8",
    )
    pairs = load_annotations(pairs_path)
    manifest = load_manifest(manifest_path)
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("empty" in e for e in report.errors)


def test_audit_invalid_box_fails(tmp_path: Path) -> None:
    bad = _pair("p1", "g1", before_classes=[0], after_classes=[0],
                statuses=["moved"])
    bad["manual_before_objects"][0]["bbox_xyxy"] = [100.0, 100.0, 50.0, 50.0]  # inverted
    records = [bad,
               _pair("p2", "g2", before_classes=list(REQUIRED_CLASS_IDS),
                     after_classes=list(REQUIRED_CLASS_IDS),
                     statuses=["added", "removed", "moved", "unchanged", "moved", "moved"])]
    pairs, manifest = _write(tmp_path, records,
                              {"train": ["p1"], "val": ["p2"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("box" in e.lower() or "positive area" in e.lower()
               for e in report.errors)


def test_audit_duplicate_class_in_same_image_fails(tmp_path: Path) -> None:
    bad = _pair("p1", "g1", before_classes=[0], after_classes=[0],
                statuses=["moved"])
    bad["manual_before_objects"].append(
        {"object_id": "b0_dup", "class_id": 0, "bbox_xyxy": [60.0, 60.0, 100.0, 100.0]}
    )
    records = [bad,
               _pair("p2", "g2", before_classes=list(REQUIRED_CLASS_IDS),
                     after_classes=list(REQUIRED_CLASS_IDS),
                     statuses=["added", "removed", "moved", "unchanged", "moved", "moved"])]
    pairs, manifest = _write(tmp_path, records,
                              {"train": ["p1"], "val": ["p2"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("duplicate class_id" in e for e in report.errors)


def test_audit_unknown_correspondence_fails(tmp_path: Path) -> None:
    bad = _pair("p1", "g1", before_classes=[0], after_classes=[0],
                statuses=["moved"])
    bad["correspondences"] = [
        {"before_object_id": "ghost", "after_object_id": "a0", "class_id": 0}
    ]
    records = [bad,
               _pair("p2", "g2", before_classes=list(REQUIRED_CLASS_IDS),
                     after_classes=list(REQUIRED_CLASS_IDS),
                     statuses=["added", "removed", "moved", "unchanged", "moved", "moved"])]
    pairs, manifest = _write(tmp_path, records,
                              {"train": ["p1"], "val": ["p2"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("correspondence" in e for e in report.errors)


def test_audit_capture_group_crosses_splits_fails(tmp_path: Path) -> None:
    records = [
        _pair("p1", "g_shared", before_classes=list(REQUIRED_CLASS_IDS),
              after_classes=list(REQUIRED_CLASS_IDS),
              statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
              no_change=True),
        _pair("p2", "g_shared", before_classes=list(REQUIRED_CLASS_IDS),
              after_classes=list(REQUIRED_CLASS_IDS),
              statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
              no_change=True),
    ]
    pairs, manifest = _write(tmp_path, records,
                              {"train": ["p1"], "val": ["p2"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("capture_group_id" in e and "crosses splits" in e
               for e in report.errors)


def test_audit_shared_image_across_splits_fails(tmp_path: Path) -> None:
    p1 = _pair("p1", "g1", before_classes=list(REQUIRED_CLASS_IDS),
               after_classes=list(REQUIRED_CLASS_IDS),
               statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
               no_change=True, before_image_id="IMG_SHARED")
    p2 = _pair("p2", "g2", before_classes=list(REQUIRED_CLASS_IDS),
               after_classes=list(REQUIRED_CLASS_IDS),
               statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
               no_change=True, before_image_id="IMG_SHARED")
    pairs, manifest = _write(tmp_path, [p1, p2],
                              {"train": ["p1"], "val": ["p2"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("image_id" in e and "shared across splits" in e
               for e in report.errors)


def test_audit_missing_class_in_split_fails(tmp_path: Path) -> None:
    # val is missing class 5
    p1 = _pair("p1", "g1", before_classes=[0, 1, 2, 3, 4],
               after_classes=[0, 1, 2, 3, 4],
               statuses=["added", "removed", "moved", "unchanged", "moved"],
               no_change=True)
    p2 = _pair("p2", "g2", before_classes=list(REQUIRED_CLASS_IDS),
               after_classes=list(REQUIRED_CLASS_IDS),
               statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
               no_change=True)
    p3 = _pair("p3", "g3", before_classes=list(REQUIRED_CLASS_IDS),
               after_classes=list(REQUIRED_CLASS_IDS),
               statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
               no_change=True)
    pairs, manifest = _write(tmp_path, [p1, p2, p3],
                              {"train": ["p2"], "val": ["p1"], "test": ["p3"]})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("val split missing classes" in e for e in report.errors)


def test_audit_missing_status_in_test_fails(tmp_path: Path) -> None:
    # test pair only has moved status
    p_test = _pair("p_test", "g_test", before_classes=list(REQUIRED_CLASS_IDS),
                   after_classes=list(REQUIRED_CLASS_IDS),
                   statuses=["moved"] * 6, no_change=True)
    p_val = _pair("p_val", "g_val", before_classes=list(REQUIRED_CLASS_IDS),
                  after_classes=list(REQUIRED_CLASS_IDS),
                  statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
                  no_change=True)
    p_train = _pair("p_train", "g_train", before_classes=list(REQUIRED_CLASS_IDS),
                    after_classes=list(REQUIRED_CLASS_IDS),
                    statuses=["added", "removed", "moved", "unchanged", "moved", "moved"])
    pairs, manifest = _write(tmp_path, [p_test, p_val, p_train],
                              {"train": ["p_train"], "val": ["p_val"], "test": ["p_test"]})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("test split missing statuses" in e for e in report.errors)


def test_audit_zero_no_change_in_val_fails(tmp_path: Path) -> None:
    p_val = _pair("p_val", "g_val", before_classes=list(REQUIRED_CLASS_IDS),
                  after_classes=list(REQUIRED_CLASS_IDS),
                  statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
                  no_change=False)
    p_train = _pair("p_train", "g_train", before_classes=list(REQUIRED_CLASS_IDS),
                    after_classes=list(REQUIRED_CLASS_IDS),
                    statuses=["added", "removed", "moved", "unchanged", "moved", "moved"])
    p_test = _pair("p_test", "g_test", before_classes=list(REQUIRED_CLASS_IDS),
                   after_classes=list(REQUIRED_CLASS_IDS),
                   statuses=["added", "removed", "moved", "unchanged", "moved", "moved"],
                   no_change=True)
    pairs, manifest = _write(tmp_path, [p_val, p_train, p_test],
                              {"train": ["p_train"], "val": ["p_val"], "test": ["p_test"]})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("val split has zero independent no-change" in e
               for e in report.errors)
