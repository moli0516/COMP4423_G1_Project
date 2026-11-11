"""Unit tests for dataset_splitter.split — whole-group assignment."""

from __future__ import annotations

import pytest

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest
from deskcheck.data.dataset_splitter import split


def _pair(
    pair_id: str,
    group_id: str,
    *,
    before_classes: list[int] | None = None,
    after_classes: list[int] | None = None,
    statuses: list[str] | None = None,
    no_change: bool = False,
    before_image_id: str | None = None,
    after_image_id: str | None = None,
) -> AnnotatedPair:
    before_classes = before_classes if before_classes is not None else [0]
    after_classes = after_classes if after_classes is not None else [0]
    statuses = statuses if statuses is not None else ["moved"]
    before_objects = tuple(
        {"object_id": f"b{cls}", "class_id": cls, "bbox_xyxy": [10, 10, 50, 50]}
        for cls in before_classes
    )
    after_objects = tuple(
        {"object_id": f"a{cls}", "class_id": cls, "bbox_xyxy": [20, 20, 60, 60]}
        for cls in after_classes
    )
    events = tuple(
        {
            "class_id": cls,
            "status": status,
            "before_object_id": f"b{cls}" if cls in before_classes else None,
            "after_object_id": f"a{cls}" if cls in after_classes else None,
        }
        for cls, status in zip(
            sorted(set(before_classes) | set(after_classes)), statuses
        )
    )
    return AnnotatedPair(
        pair_id=pair_id,
        capture_group_id=group_id,
        before_image_id=before_image_id or f"{pair_id}_b",
        after_image_id=after_image_id or f"{pair_id}_a",
        before_image_path=f"raw/{pair_id}_b.jpg",
        after_image_path=f"raw/{pair_id}_a.jpg",
        width=640,
        height=480,
        rules_version="1.0",
        independently_captured_no_change=no_change,
        manual_before_objects=before_objects,
        manual_after_objects=after_objects,
        correspondences=(),
        events=events,
    )


ALL_CLASSES = [0, 1, 2, 3, 4, 5]
ALL_STATUSES = ["added", "removed", "moved", "unchanged", "moved", "moved"]


def _healthy_pairs(count: int = 10) -> list[AnnotatedPair]:
    pairs: list[AnnotatedPair] = []
    for i in range(count):
        pid = f"p{i}"
        gid = f"g{i}"
        no_change = i in (3, 4, 7, 8)
        pairs.append(_pair(
            pid, gid,
            before_classes=ALL_CLASSES,
            after_classes=ALL_CLASSES,
            statuses=ALL_STATUSES,
            no_change=no_change,
        ))
    return pairs


def test_split_assigns_every_pair_exactly_once() -> None:
    pairs = _healthy_pairs()
    manifest = split(pairs, seed=42)
    all_ids = [pid for ids in manifest.splits.values() for pid in ids]
    assert sorted(all_ids) == sorted(p.pair_id for p in pairs)


def test_split_is_deterministic_with_seed() -> None:
    pairs = _healthy_pairs()
    m1 = split(pairs, seed=42)
    m2 = split(pairs, seed=42)
    assert m1.splits == m2.splits


def test_split_differs_with_different_seed() -> None:
    pairs = _healthy_pairs()
    m1 = split(pairs, seed=42)
    m2 = split(pairs, seed=99)
    assert m1.splits != m2.splits


def test_split_keeps_groups_whole() -> None:
    # 4 buckets: g_shared (p1+p2), g_other (p3), g_indep (p4)
    # Guarantees at least 3 buckets so train/val/test can all be non-empty.
    pairs = [
        _pair("p1", "g_shared",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True),
        _pair("p2", "g_shared",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True),
        _pair("p3", "g_other",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True),
        _pair("p4", "g_indep",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True),
    ]
    manifest = split(pairs, seed=42)
    for ids in manifest.splits.values():
        if "p1" in ids:
            assert "p2" in ids, "p1 and p2 share a group but landed in different splits"


def test_split_keeps_shared_image_groups_whole() -> None:
    # p1 and p2 share IMG2, so they merge into one bucket.
    # Provide p3 and p4 as independent buckets to keep train/val/test non-empty.
    pairs = [
        _pair("p1", "g1",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True,
              before_image_id="IMG1", after_image_id="IMG2"),
        _pair("p2", "g2",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True,
              before_image_id="IMG2", after_image_id="IMG3"),
        _pair("p3", "g3",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True),
        _pair("p4", "g4",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=True),
    ]
    manifest = split(pairs, seed=42)
    for ids in manifest.splits.values():
        if "p1" in ids:
            assert "p2" in ids, "p1 and p2 share IMG2 but landed in different splits"


def test_split_guarantees_no_change_in_val_and_test() -> None:
    pairs = _healthy_pairs()
    manifest = split(pairs, seed=42)
    for split_name in ("val", "test"):
        split_ids = set(manifest.splits[split_name])
        no_change_in_split = {
            p.pair_id for p in pairs
            if p.pair_id in split_ids and p.independently_captured_no_change
        }
        assert no_change_in_split, f"{split_name} has no independent no-change pair"


def test_split_guarantees_all_statuses_in_val_and_test() -> None:
    pairs = _healthy_pairs()
    manifest = split(pairs, seed=42)
    required = {"added", "removed", "moved", "unchanged"}
    for split_name in ("val", "test"):
        split_ids = set(manifest.splits[split_name])
        seen: set[str] = set()
        for p in pairs:
            if p.pair_id in split_ids:
                for ev in p.events:
                    seen.add(ev["status"])
        assert required <= seen, f"{split_name} missing statuses {required - seen}"


def test_split_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        split([], seed=42)


def test_split_rejects_bad_ratio_sum() -> None:
    pairs = _healthy_pairs()
    with pytest.raises(ValueError, match="sum to 1.0"):
        split(pairs, seed=42, target_ratios={"train": 0.5, "val": 0.2, "test": 0.2})


def test_split_rejects_missing_split_name() -> None:
    pairs = _healthy_pairs()
    with pytest.raises(ValueError, match="train, val, test"):
        split(pairs, seed=42, target_ratios={"train": 0.7, "val": 0.3})


def test_split_raises_when_no_no_change_buckets() -> None:
    pairs = [
        _pair(f"p{i}", f"g{i}",
              before_classes=ALL_CLASSES, after_classes=ALL_CLASSES,
              statuses=ALL_STATUSES, no_change=False)
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="no-change"):
        split(pairs, seed=42)


def test_split_returns_split_manifest_type() -> None:
    pairs = _healthy_pairs()
    manifest = split(pairs, seed=42)
    assert isinstance(manifest, SplitManifest)
    assert manifest.seed == 42
    assert set(manifest.splits.keys()) == {"train", "val", "test"}
