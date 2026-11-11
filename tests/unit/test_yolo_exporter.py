"""Unit + integration tests for yolo_exporter.export — real image bytes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest
from deskcheck.data.yolo_exporter import export


def _make_image_file(path: Path, width: int = 20, height: int = 10) -> None:
    """Write a tiny real PNG. Uses numpy to guarantee non-trivial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = 128
        Image.fromarray(arr).save(path)
    except ImportError:
        # Fallback: write a 1x1 valid PNG bytes (still a real image file)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00"
            b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        path.write_bytes(png_bytes)


def _pair(
    pair_id: str, group_id: str, *, before_objs, after_objs,
    before_path: Path, after_path: Path,
    width: int = 20, height: int = 10,
    before_image_id: str | None = None,
    after_image_id: str | None = None,
) -> AnnotatedPair:
    return AnnotatedPair(
        pair_id=pair_id,
        capture_group_id=group_id,
        before_image_id=before_image_id or f"{pair_id}_b",
        after_image_id=after_image_id or f"{pair_id}_a",
        before_image_path=str(before_path),
        after_image_path=str(after_path),
        width=width, height=height,
        rules_version="1.0",
        independently_captured_no_change=False,
        manual_before_objects=tuple(
            {"object_id": f"b_{i}", "class_id": cls, "bbox_xyxy": box}
            for i, (cls, box) in enumerate(before_objs)
        ),
        manual_after_objects=tuple(
            {"object_id": f"a_{i}", "class_id": cls, "bbox_xyxy": box}
            for i, (cls, box) in enumerate(after_objs)
        ),
        correspondences=(),
        events=(),
    )


def _manifest(splits: dict[str, list[str]]) -> SplitManifest:
    return SplitManifest(
        schema_version="1.0", seed=42, dataset_release="v1",
        splits={k: tuple(v) for k, v in splits.items()}, coverage={},
    )


def _make_pair_with_images(tmp_path: Path, pair_id: str, group_id: str) -> AnnotatedPair:
    b = tmp_path / "raw" / group_id / f"{pair_id}_before.png"
    a = tmp_path / "raw" / group_id / f"{pair_id}_after.png"
    _make_image_file(b)
    _make_image_file(a)
    return _pair(
        pair_id, group_id,
        before_objs=[(0, [1.0, 1.0, 10.0, 8.0])],
        after_objs=[(0, [2.0, 2.0, 11.0, 9.0])],
        before_path=b, after_path=a,
    )


# ---------- 1. materialize images ----------


def test_export_copies_image_bytes_into_images_split(tmp_path: Path) -> None:
    pair = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"

    summary = export({"p1": pair}, manifest, out)

    assert summary.images_written == 2
    assert (out / "images" / "train" / "p1__before.png").exists()
    assert (out / "images" / "train" / "p1__after.png").exists()
    # Copied bytes are actual bytes, not empty files
    assert (out / "images" / "train" / "p1__before.png").stat().st_size > 0


def test_export_labels_match_image_stems(tmp_path: Path) -> None:
    pair = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"
    export({"p1": pair}, manifest, out)

    images = sorted(p.stem for p in (out / "images" / "train").iterdir())
    labels = sorted(p.stem for p in (out / "labels" / "train").iterdir())
    assert images == labels == ["p1__after", "p1__before"]


def test_export_dataset_yaml_points_at_valid_paths(tmp_path: Path) -> None:
    pair = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"
    export({"p1": pair}, manifest, out)

    yaml_text = (out / "dataset.yaml").read_text(encoding="utf-8")
    assert "nc: 6" in yaml_text
    assert "train: images/train" in yaml_text
    assert (out / "images" / "train").is_dir()


# ---------- 2. no filename collisions ----------


def test_export_unique_keys_across_groups(tmp_path: Path) -> None:
    """Two pairs in different groups both have 'before.png' source filename.
    Export must produce distinct files."""
    p1 = _make_pair_with_images(tmp_path, "p1", "g001")
    p2 = _make_pair_with_images(tmp_path, "p2", "g002")
    manifest = _manifest({"train": ["p1", "p2"], "val": [], "test": []})
    out = tmp_path / "export"
    export({"p1": p1, "p2": p2}, manifest, out)

    train_images = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert len(train_images) == 4
    assert "p1__before.png" in train_images
    assert "p2__before.png" in train_images
    assert len(set(train_images)) == 4


# ---------- 3. shared-image consistency ----------


def test_export_rejects_inconsistent_shared_image(tmp_path: Path) -> None:
    """Same image_id, different paths/dims -> refuse."""
    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    p2 = _make_pair_with_images(tmp_path, "p2", "g2")
    # Force p2.before_image_id to collide with p1.after_image_id
    p2 = AnnotatedPair(
        pair_id=p2.pair_id,
        capture_group_id=p2.capture_group_id,
        before_image_id=p1.after_image_id,   # collision
        after_image_id=p2.after_image_id,
        before_image_path=p2.before_image_path,  # different path
        after_image_path=p2.after_image_path,
        width=p2.width, height=p2.height,
        rules_version=p2.rules_version,
        independently_captured_no_change=False,
        manual_before_objects=p2.manual_before_objects,
        manual_after_objects=p2.manual_after_objects,
        correspondences=(), events=(),
    )
    manifest = _manifest({"train": ["p1", "p2"], "val": [], "test": []})
    with pytest.raises(ValueError, match="conflicting representations"):
        export({"p1": p1, "p2": p2}, manifest, tmp_path / "export")


def test_export_allows_consistent_shared_image(tmp_path: Path) -> None:
    """Same image_id, same path/dims/annotations -> dedupe silently."""
    shared = tmp_path / "raw" / "shared.png"
    _make_image_file(shared)
    p1 = _pair("p1", "g1",
               before_objs=[(0, [1.0, 1.0, 10.0, 8.0])],
               after_objs=[(1, [2.0, 2.0, 11.0, 9.0])],
               before_path=shared, after_path=shared,
               before_image_id="IMG_X", after_image_id="p1_a")
    p2 = _pair("p2", "g2",
               before_objs=[(0, [1.0, 1.0, 10.0, 8.0])],
               after_objs=[(1, [2.0, 2.0, 11.0, 9.0])],
               before_path=shared, after_path=shared,
               before_image_id="IMG_X", after_image_id="p2_a")
    manifest = _manifest({"train": ["p1", "p2"], "val": [], "test": []})
    summary = export({"p1": p1, "p2": p2}, manifest, tmp_path / "export")
    # 3 unique image_ids: IMG_X, p1_a, p2_a
    assert summary.images_written == 3
    assert summary.shared_images_deduplicated == 1


# ---------- 4. overwrite protection ----------


def test_export_refuses_nonempty_output(tmp_path: Path) -> None:
    pair = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"
    export({"p1": pair}, manifest, out)

    with pytest.raises(FileExistsError, match="already has content"):
        export({"p1": pair}, manifest, out)


def test_export_overwrite_allowed_with_flag(tmp_path: Path) -> None:
    pair = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"
    export({"p1": pair}, manifest, out)
    # Second run with overwrite=True should succeed
    export({"p1": pair}, manifest, out, overwrite=True)


# ---------- 5. missing file error ----------


def test_export_fails_if_image_bytes_missing(tmp_path: Path) -> None:
    pair = _pair(
        "p1", "g1",
        before_objs=[(0, [1.0, 1.0, 10.0, 8.0])],
        after_objs=[],
        before_path=tmp_path / "does_not_exist.png",
        after_path=tmp_path / "does_not_exist2.png",
    )
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    with pytest.raises(FileNotFoundError, match="image file missing"):
        export({"p1": pair}, manifest, tmp_path / "export")


# ---------- 6. class coverage across splits ----------


def test_export_keeps_splits_separate(tmp_path: Path) -> None:
    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    p2 = _make_pair_with_images(tmp_path, "p2", "g2")
    manifest = _manifest({"train": ["p1"], "val": ["p2"], "test": []})
    out = tmp_path / "export"
    export({"p1": p1, "p2": p2}, manifest, out)

    assert any((out / "images" / "train").iterdir())
    assert any((out / "images" / "val").iterdir())
    assert not any((out / "images" / "test").iterdir())
