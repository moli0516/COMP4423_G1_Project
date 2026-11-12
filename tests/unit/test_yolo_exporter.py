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


# ---------- overwrite removes stale files ----------


def test_overwrite_removes_stale_files_from_previous_export(tmp_path: Path) -> None:
    """Re-export with fewer pairs must not leave old images/labels behind."""
    from deskcheck.data.yolo_exporter import ExportSummary as _ES  # noqa: F401

    # First export: 3 pairs
    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    p2 = _make_pair_with_images(tmp_path, "p2", "g2")
    p3 = _make_pair_with_images(tmp_path, "p3", "g3")
    manifest_a = _manifest({"train": ["p1", "p2", "p3"], "val": [], "test": []})
    out = tmp_path / "export"
    export({"p1": p1, "p2": p2, "p3": p3}, manifest_a, out)

    # Confirm first export has 6 images (3 pairs × 2 sides)
    first_images = list((out / "images" / "train").iterdir())
    assert len(first_images) == 6

    # Second export with overwrite=True, only 1 pair
    manifest_b = _manifest({"train": ["p1"], "val": [], "test": []})
    export({"p1": p1}, manifest_b, out, overwrite=True)

    # Only p1's 2 images should exist now
    second_images = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert second_images == ["p1__after.png", "p1__before.png"]

    second_labels = sorted(p.name for p in (out / "labels" / "train").iterdir())
    assert second_labels == ["p1__after.txt", "p1__before.txt"]


def test_overwrite_removes_stale_split_directories(tmp_path: Path) -> None:
    """If a split existed before and no longer does, its dir must be gone."""
    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    p2 = _make_pair_with_images(tmp_path, "p2", "g2")
    manifest_a = _manifest({"train": ["p1"], "val": ["p2"], "test": []})
    out = tmp_path / "export"
    export({"p1": p1, "p2": p2}, manifest_a, out)
    assert (out / "images" / "val").exists()

    # Re-export with everything in train
    manifest_b = _manifest({"train": ["p1", "p2"], "val": [], "test": []})
    export({"p1": p1, "p2": p2}, manifest_b, out, overwrite=True)

    # val/ labels/ image dirs should be gone or empty (no files)
    val_imgs = out / "images" / "val"
    if val_imgs.exists():
        assert not any(val_imgs.iterdir()), "val/ should be empty after overwrite"


def test_refuse_overwrite_without_flag(tmp_path: Path) -> None:
    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"
    export({"p1": p1}, manifest, out)

    import pytest as _pytest
    with _pytest.raises(FileExistsError, match="already has content"):
        export({"p1": p1}, manifest, out, overwrite=False)


def test_validator_rejects_cross_split_shared_image_id(tmp_path: Path) -> None:
    """A shared image_id across splits must fail the audit.

    This exercises the check that the export CLI now relies on.
    """
    from deskcheck.data.dataset_validator import audit

    p_train = _make_pair_with_images(tmp_path, "p_tr", "g_tr")
    p_val = _make_pair_with_images(tmp_path, "p_va", "g_va")
    # Force p_val.before_image_id to collide with p_train.after_image_id
    p_val = AnnotatedPair(
        pair_id=p_val.pair_id,
        capture_group_id=p_val.capture_group_id,
        before_image_id=p_train.after_image_id,
        after_image_id=p_val.after_image_id,
        before_image_path=p_val.before_image_path,
        after_image_path=p_val.after_image_path,
        width=p_val.width, height=p_val.height,
        rules_version=p_val.rules_version,
        independently_captured_no_change=False,
        manual_before_objects=p_val.manual_before_objects,
        manual_after_objects=p_val.manual_after_objects,
        correspondences=(), events=(),
    )
    pairs = {"p_tr": p_train, "p_va": p_val}
    manifest = _manifest({"train": ["p_tr"], "val": ["p_va"], "test": []})
    report = audit(pairs, manifest)
    assert not report.ok
    assert any("shared across splits" in e for e in report.errors)


def test_export_cli_blocked_by_failed_audit(tmp_path: Path) -> None:
    """The export CLI must exit 2 if the audit fails, without writing output."""
    import json
    import subprocess
    import sys

    # Build a set of files that fail audit (shared image across splits)
    p_train = _make_pair_with_images(tmp_path, "p_tr", "g_tr")
    p_val = _make_pair_with_images(tmp_path, "p_va", "g_va")
    p_val = AnnotatedPair(
        pair_id=p_val.pair_id,
        capture_group_id=p_val.capture_group_id,
        before_image_id=p_train.after_image_id,
        after_image_id=p_val.after_image_id,
        before_image_path=p_val.before_image_path,
        after_image_path=p_val.after_image_path,
        width=p_val.width, height=p_val.height,
        rules_version=p_val.rules_version,
        independently_captured_no_change=False,
        manual_before_objects=p_val.manual_before_objects,
        manual_after_objects=p_val.manual_after_objects,
        correspondences=(), events=(),
    )

    # Write annotations JSONL
    ann = tmp_path / "pairs.jsonl"
    def _serialize(p):
        return {
            "schema_version": "1.0",
            "pair_id": p.pair_id,
            "capture_group_id": p.capture_group_id,
            "before_image_id": p.before_image_id,
            "after_image_id": p.after_image_id,
            "before_image_path": p.before_image_path,
            "after_image_path": p.after_image_path,
            "width": p.width, "height": p.height,
            "rules_version": p.rules_version,
            "independently_captured_no_change": p.independently_captured_no_change,
            "manual_before_objects": list(p.manual_before_objects),
            "manual_after_objects": list(p.manual_after_objects),
            "correspondences": list(p.correspondences),
            "events": list(p.events),
        }
    with ann.open("w", encoding="utf-8") as fh:
        for p in (p_train, p_val):
            fh.write(json.dumps(_serialize(p)) + "\n")

    # Write manifest
    mf = tmp_path / "splits.json"
    mf.write_text(json.dumps({
        "schema_version": "1.0", "seed": 42, "dataset_release": "v1",
        "splits": {"train": ["p_tr"], "val": ["p_va"], "test": []},
    }), encoding="utf-8")

    output = tmp_path / "export"

    result = subprocess.run(
        [sys.executable, "scripts/export_yolo_dataset.py",
         "--annotations", str(ann),
         "--manifest", str(mf),
         "--output", str(output)],
        check=False,
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    assert result.returncode == 2, f"expected exit 2, got {result.returncode}: {result.stderr}"
    assert "audit failed" in result.stderr
    assert not output.exists() or not any(output.iterdir()), \
        "output directory should not exist or be empty after failed audit"


# ---------- atomic publish ----------


def test_export_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    """If a mid-export copy fails, the target output_dir must be untouched."""
    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    p2 = _make_pair_with_images(tmp_path, "p2", "g2")
    # Corrupt p2's after image path to a non-existent file
    p2 = AnnotatedPair(
        pair_id=p2.pair_id,
        capture_group_id=p2.capture_group_id,
        before_image_id=p2.before_image_id,
        after_image_id=p2.after_image_id,
        before_image_path=p2.before_image_path,
        after_image_path=str(tmp_path / "does_not_exist_after.png"),
        width=p2.width, height=p2.height,
        rules_version=p2.rules_version,
        independently_captured_no_change=False,
        manual_before_objects=p2.manual_before_objects,
        manual_after_objects=p2.manual_after_objects,
        correspondences=(), events=(),
    )
    manifest = _manifest({"train": ["p1", "p2"], "val": [], "test": []})
    out = tmp_path / "export"

    with pytest.raises(FileNotFoundError):
        export({"p1": p1, "p2": p2}, manifest, out)

    # Output dir must not exist (or must be empty) after failure.
    assert not out.exists() or not any(out.iterdir()), (
        f"partial export remained at {out}"
    )
    # No temp dirs or old dirs left behind in the parent.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".export")]
    assert not leftovers, f"temp dirs leaked: {leftovers}"


def test_export_overwrite_swap_preserves_new_content_on_success(tmp_path: Path) -> None:
    """When overwrite=True and export succeeds, the new content replaces the old."""
    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"

    # First export
    export({"p1": p1}, manifest, out)
    first_files = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert "p1__before.png" in first_files

    # Re-export with the same data (overwrite=True) — should succeed cleanly.
    export({"p1": p1}, manifest, out, overwrite=True)
    second_files = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert second_files == first_files

    # No stray .old/ .tmp dirs remain.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".export")]
    assert not leftovers, f"temp/old dirs leaked: {leftovers}"


def test_export_atomic_swap_no_partial_on_write_failure(tmp_path: Path) -> None:
    """If a label write fails midway, target output_dir must remain untouched."""
    from unittest.mock import patch

    p1 = _make_pair_with_images(tmp_path, "p1", "g1")
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "export"

    # Pre-populate out with a valid first export.
    export({"p1": p1}, manifest, out)
    old_image_files = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert old_image_files

    # Force _write_into to fail AFTER the existing output is in place.
    import deskcheck.data.yolo_exporter as mod
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated disk failure")
    with (
        patch.object(mod, "_write_into", _boom),
        pytest.raises(RuntimeError, match="simulated disk failure"),
    ):
        export({"p1": p1}, manifest, out, overwrite=True)

    # Original output must still be intact.
    assert out.exists()
    new_image_files = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert new_image_files == old_image_files, (
        f"output_dir was modified despite write failure: {new_image_files}"
    )

    # No temp or old dirs remain.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".export")]
    assert not leftovers, f"temp/old dirs leaked: {leftovers}"
