"""Unit tests for dataset_validator.audit."""

from __future__ import annotations

import json
from pathlib import Path

from deskcheck.data.dataset_catalog import load_annotations, load_manifest
from deskcheck.data.dataset_validator import audit


def _pair(
    pair_id: str,
    group_id: str,
    *,
    before_classes: list[int],
    after_classes: list[int],
    statuses_by_class: dict[int, str],
    no_change: bool = False,
    before_image_id: str | None = None,
    after_image_id: str | None = None,
) -> dict:
    """Build a semantically valid pair."""
    events = []
    for cls, status in statuses_by_class.items():
        if status == "added":
            events.append({"class_id": cls, "status": "added",
                           "before_object_id": None,
                           "after_object_id": f"a_{cls}"})
        elif status == "removed":
            events.append({"class_id": cls, "status": "removed",
                           "before_object_id": f"b_{cls}",
                           "after_object_id": None})
        else:
            events.append({"class_id": cls, "status": status,
                           "before_object_id": f"b_{cls}",
                           "after_object_id": f"a_{cls}"})

    correspondences = [
        {"before_object_id": f"b_{cls}", "after_object_id": f"a_{cls}",
         "class_id": cls}
        for cls, status in statuses_by_class.items()
        if status in ("moved", "unchanged")
    ]

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
        "manual_before_objects": [
            {"object_id": f"b_{cls}", "class_id": cls,
             "bbox_xyxy": [float(10 + cls * 80), 10.0,
                           float(60 + cls * 80), 60.0]}
            for cls in before_classes
        ],
        "manual_after_objects": [
            {"object_id": f"a_{cls}", "class_id": cls,
             "bbox_xyxy": [float(10 + cls * 80), 10.0,
                           float(60 + cls * 80), 60.0]}
            for cls in after_classes
        ],
        "correspondences": correspondences,
        "events": events,
    }


def _write(tmp_path: Path, records: list[dict], splits: dict[str, list[str]]) -> tuple:
    pairs_path = tmp_path / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0", "seed": 42, "dataset_release": "v1",
                    "splits": splits}),
        encoding="utf-8",
    )
    return load_annotations(pairs_path), load_manifest(manifest_path)


def _make_valid_pair(pair_id: str, group_id: str, *, no_change: bool = False) -> dict:
    """One valid pair covering all 4 statuses + all 6 classes."""
    if no_change:
        return _pair(
            pair_id, group_id,
            before_classes=[0, 1, 2, 3, 4, 5],
            after_classes=[0, 1, 2, 3, 4, 5],
            statuses_by_class={c: "unchanged" for c in range(6)},
            no_change=True,
        )
    return _pair(
        pair_id, group_id,
        before_classes=[0, 1, 2, 3, 4],
        after_classes=[0, 2, 3, 4, 5],
        statuses_by_class={0: "unchanged", 1: "removed", 2: "moved",
                            3: "unchanged", 4: "unchanged", 5: "added"},
    )


def _minimal_healthy_dataset(tmp_path: Path) -> tuple:
    records = [
        _make_valid_pair("p1", "g1"),
        _make_valid_pair("p2", "g2"),
        _make_valid_pair("p3", "g3", no_change=True),
        _make_valid_pair("p4", "g4"),
        _make_valid_pair("p5", "g5"),
        _make_valid_pair("p6", "g6", no_change=True),
    ]
    splits = {"train": ["p1", "p2"], "val": ["p3", "p4"], "test": ["p5", "p6"]}
    return _write(tmp_path, records, splits)


# ---------- happy paths ----------


def test_audit_healthy_dataset_passes(tmp_path: Path) -> None:
    pairs, manifest = _minimal_healthy_dataset(tmp_path)
    report = audit(pairs, manifest)
    assert report.ok, report.errors
    assert report.independent_no_change_counts["val"] >= 1
    assert report.independent_no_change_counts["test"] >= 1


def test_audit_warns_when_tau_missing(tmp_path: Path) -> None:
    pairs, manifest = _minimal_healthy_dataset(tmp_path)
    report = audit(pairs, manifest)
    assert any("movement threshold not supplied" in w for w in report.warnings)


# ---------- semantic negative tests ----------


def test_audit_rejects_added_with_before_present(tmp_path: Path) -> None:
    bad = _pair("bad", "gb", before_classes=[0], after_classes=[0],
                 statuses_by_class={0: "added"})
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("'added' for class 0 requires absent-before" in e
               for e in report.errors)


def test_audit_rejects_removed_with_after_present(tmp_path: Path) -> None:
    bad = _pair("bad", "gb", before_classes=[0], after_classes=[0],
                 statuses_by_class={0: "removed"})
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("'removed' for class 0 requires present-before" in e
               for e in report.errors)


def test_audit_rejects_moved_with_one_side_missing(tmp_path: Path) -> None:
    bad = _pair("bad", "gb", before_classes=[0], after_classes=[],
                 statuses_by_class={0: "moved"})
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("'moved' for class 0 requires both sides present" in e
               for e in report.errors)


def test_audit_rejects_change_event_in_no_change_pair(tmp_path: Path) -> None:
    bad = _pair("bad", "gb", before_classes=[0], after_classes=[0, 1],
                 statuses_by_class={0: "unchanged", 1: "added"},
                 no_change=True)
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("independent no-change pair contains change events" in e
               for e in report.errors)


def test_audit_rejects_missing_event_for_present_class(tmp_path: Path) -> None:
    bad = _pair("bad", "gb", before_classes=[0, 1], after_classes=[0, 1],
                 statuses_by_class={0: "unchanged"})
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("no event" in e for e in report.errors)


def test_audit_rejects_duplicate_event(tmp_path: Path) -> None:
    bad = _pair("bad", "gb", before_classes=[0], after_classes=[0],
                 statuses_by_class={0: "unchanged"})
    bad["events"].append(dict(bad["events"][0]))
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("duplicate event for class_id 0" in e for e in report.errors)


def test_audit_rejects_correspondence_class_mismatch(tmp_path: Path) -> None:
    bad = _pair("bad", "gb", before_classes=[0], after_classes=[0],
                 statuses_by_class={0: "unchanged"})
    bad["correspondences"] = [
        {"before_object_id": "b_0", "after_object_id": "a_0", "class_id": 3}
    ]
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("class_id 3 disagrees" in e for e in report.errors)


def test_audit_rejects_invalid_class_id(tmp_path: Path) -> None:
    bad = _make_valid_pair("bad", "gb")
    bad["manual_before_objects"].append(
        {"object_id": "b_99", "class_id": 99,
         "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]})
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("class_id 99 not in" in e for e in report.errors)


def test_audit_rejects_duplicate_object_id(tmp_path: Path) -> None:
    bad = _make_valid_pair("bad", "gb")
    bad["manual_before_objects"].append(
        {"object_id": "b_0", "class_id": 3,
         "bbox_xyxy": [200.0, 0.0, 210.0, 10.0]})
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["bad"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("duplicate object_id" in e for e in report.errors)


# ---------- pre-existing negative tests ----------


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
    bad = _make_valid_pair("p1", "g1")
    bad["manual_before_objects"][0]["bbox_xyxy"] = [100.0, 100.0, 50.0, 50.0]
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["p1"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok


def test_audit_duplicate_class_in_same_image_fails(tmp_path: Path) -> None:
    bad = _make_valid_pair("p1", "g1")
    bad["manual_before_objects"].append(
        {"object_id": "extra", "class_id": 0,
         "bbox_xyxy": [200.0, 200.0, 250.0, 250.0]})
    pairs, manifest = _write(tmp_path, [bad],
                              {"train": ["p1"], "val": [], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("duplicate class_id" in e for e in report.errors)


def test_audit_capture_group_crosses_splits_fails(tmp_path: Path) -> None:
    p1 = _make_valid_pair("p1", "g_shared")
    p2 = _make_valid_pair("p2", "g_shared")
    pairs, manifest = _write(tmp_path, [p1, p2],
                              {"train": ["p1"], "val": ["p2"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("capture_group_id" in e and "crosses splits" in e
               for e in report.errors)


def test_audit_shared_image_across_splits_fails(tmp_path: Path) -> None:
    p1 = _make_valid_pair("p1", "g1", no_change=True)
    p1["before_image_id"] = "IMG_SHARED"
    p2 = _make_valid_pair("p2", "g2", no_change=True)
    p2["before_image_id"] = "IMG_SHARED"
    pairs, manifest = _write(tmp_path, [p1, p2],
                              {"train": ["p1"], "val": ["p2"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("image_id" in e and "shared across splits" in e
               for e in report.errors)


def test_audit_missing_class_in_split_fails(tmp_path: Path) -> None:
    p_val = _pair("p_val", "gv",
                   before_classes=[0, 1, 2, 3, 4],
                   after_classes=[0, 2, 3, 4],
                   statuses_by_class={0: "unchanged", 1: "removed", 2: "moved",
                                       3: "unchanged", 4: "unchanged"})
    p_train = _make_valid_pair("p_train", "gt")
    p_test = _make_valid_pair("p_test", "gte")
    pairs, manifest = _write(tmp_path, [p_val, p_train, p_test],
                              {"train": ["p_train"], "val": ["p_val"],
                               "test": ["p_test"]})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("val split missing classes" in e for e in report.errors)


def test_audit_zero_no_change_in_val_fails(tmp_path: Path) -> None:
    p_val = _make_valid_pair("p_val", "gv")
    p_train = _make_valid_pair("p_train", "gt")
    p_test = _make_valid_pair("p_test", "gte", no_change=True)
    pairs, manifest = _write(tmp_path, [p_val, p_train, p_test],
                              {"train": ["p_train"], "val": ["p_val"],
                               "test": ["p_test"]})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("val split has zero independent no-change" in e
               for e in report.errors)


def test_audit_missing_status_in_test_fails(tmp_path: Path) -> None:
    p_test = _pair("p_test", "gte", before_classes=[0], after_classes=[0],
                    statuses_by_class={0: "unchanged"}, no_change=True)
    p_val = _make_valid_pair("p_val", "gv", no_change=True)
    p_train = _make_valid_pair("p_train", "gt")
    pairs, manifest = _write(tmp_path, [p_test, p_val, p_train],
                              {"train": ["p_train"], "val": ["p_val"],
                               "test": ["p_test"]})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("test split missing statuses" in e for e in report.errors)
