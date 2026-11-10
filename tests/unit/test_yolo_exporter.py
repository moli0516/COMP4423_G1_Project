"""Unit tests for yolo_exporter.export — label format, dedup, split isolation."""

from __future__ import annotations

from pathlib import Path

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest
from deskcheck.data.yolo_exporter import DEFAULT_CLASS_NAMES, export


def _pair(
    pair_id: str,
    group_id: str,
    *,
    before_objs: list[tuple[int, list[float]]],
    after_objs: list[tuple[int, list[float]]],
    width: int = 640,
    height: int = 480,
    before_image_id: str | None = None,
    after_image_id: str | None = None,
) -> AnnotatedPair:
    before_objects = tuple(
        {"object_id": f"b{i}", "class_id": cls, "bbox_xyxy": box}
        for i, (cls, box) in enumerate(before_objs)
    )
    after_objects = tuple(
        {"object_id": f"a{i}", "class_id": cls, "bbox_xyxy": box}
        for i, (cls, box) in enumerate(after_objs)
    )
    return AnnotatedPair(
        pair_id=pair_id,
        capture_group_id=group_id,
        before_image_id=before_image_id or f"{pair_id}_b",
        after_image_id=after_image_id or f"{pair_id}_a",
        before_image_path=f"raw/{pair_id}_b.jpg",
        after_image_path=f"raw/{pair_id}_a.jpg",
        width=width,
        height=height,
        rules_version="1.0",
        independently_captured_no_change=False,
        manual_before_objects=before_objects,
        manual_after_objects=after_objects,
        correspondences=(),
        events=(),
    )


def _manifest(splits: dict[str, list[str]]) -> SplitManifest:
    return SplitManifest(
        schema_version="1.0",
        seed=42,
        dataset_release="v1",
        splits={name: tuple(ids) for name, ids in splits.items()},
        coverage={},
    )


def _parse_label_line(line: str) -> tuple[int, list[float]]:
    parts = line.split()
    return int(parts[0]), [float(x) for x in parts[1:]]


# ---------- dataset.yaml ----------


def test_export_writes_dataset_yaml(tmp_path: Path) -> None:
    pair = _pair("p1", "g1", before_objs=[(0, [10, 10, 50, 50])],
                 after_objs=[(0, [20, 20, 60, 60])])
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    export({"p1": pair}, manifest, tmp_path / "out")
    yaml_path = tmp_path / "out" / "dataset.yaml"
    assert yaml_path.exists()
    text = yaml_path.read_text(encoding="utf-8")
    assert "nc: 6" in text
    assert "bottle" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text


# ---------- label format ----------


def test_export_label_is_normalized_center_wh(tmp_path: Path) -> None:
    # Box [100, 100, 300, 200] in 640x480 image.
    # cx = 200/640 = 0.3125, cy = 150/480 = 0.3125
    # w  = 200/640 = 0.3125, h = 100/480 = 0.208333...
    pair = _pair("p1", "g1", before_objs=[(0, [100, 100, 300, 200])],
                 after_objs=[])
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    export({"p1": pair}, manifest, tmp_path / "out")

    label_path = tmp_path / "out" / "labels" / "train" / "p1_b.txt"
    assert label_path.exists()
    lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 1
    class_id, vals = _parse_label_line(lines[0])
    assert class_id == 0
    cx, cy, w, h = vals
    assert abs(cx - 0.3125) < 1e-5
    assert abs(cy - 0.3125) < 1e-5
    assert abs(w - 0.3125) < 1e-5
    assert abs(h - (100 / 480)) < 1e-5


def test_export_preserves_class_ids(tmp_path: Path) -> None:
    pair = _pair(
        "p1", "g1",
        before_objs=[(0, [10, 10, 50, 50]), (3, [100, 100, 150, 150])],
        after_objs=[(5, [200, 200, 250, 250])],
    )
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    export({"p1": pair}, manifest, tmp_path / "out")

    before_lines = [ln for ln in (tmp_path / "out" / "labels" / "train" / "p1_b.txt")
                    .read_text(encoding="utf-8").splitlines() if ln]
    assert {int(ln.split()[0]) for ln in before_lines} == {0, 3}

    after_lines = [ln for ln in (tmp_path / "out" / "labels" / "train" / "p1_a.txt")
                   .read_text(encoding="utf-8").splitlines() if ln]
    assert {int(ln.split()[0]) for ln in after_lines} == {5}


# ---------- empty labels ----------


def test_export_empty_labels_are_valid_empty_files(tmp_path: Path) -> None:
    # Empty scene: no objects on either side.
    pair = _pair("p1", "g1", before_objs=[], after_objs=[])
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    summary = export({"p1": pair}, manifest, tmp_path / "out")

    before = (tmp_path / "out" / "labels" / "train" / "p1_b.txt").read_text()
    after = (tmp_path / "out" / "labels" / "train" / "p1_a.txt").read_text()
    assert before == ""
    assert after == ""
    assert summary.empty_labels_written == 2


# ---------- shared image dedupe ----------


def test_export_dedupes_shared_image(tmp_path: Path) -> None:
    # p1 after-image = p2 before-image (same image_id "IMG_X").
    p1 = _pair("p1", "g1",
               before_objs=[(0, [10, 10, 50, 50])],
               after_objs=[(0, [20, 20, 60, 60])],
               before_image_id="IMG_1", after_image_id="IMG_X")
    p2 = _pair("p2", "g2",
               before_objs=[(0, [30, 30, 70, 70])],
               after_objs=[(0, [40, 40, 80, 80])],
               before_image_id="IMG_X", after_image_id="IMG_2")
    manifest = _manifest({"train": ["p1", "p2"], "val": [], "test": []})
    summary = export({"p1": p1, "p2": p2}, manifest, tmp_path / "out")

    # 3 unique images total: IMG_1, IMG_X, IMG_2
    assert summary.images_written == 3
    label_files = sorted(p.name for p in (tmp_path / "out" / "labels" / "train").iterdir())
    # Exporter names label files after the image path stem, not image_id, so the YOLO
    # labels can sit alongside the image bytes during training.
    assert label_files == ["p1_a.txt", "p1_b.txt", "p2_a.txt"]


# ---------- split isolation ----------


def test_export_keeps_splits_separate(tmp_path: Path) -> None:
    p_train = _pair("p_tr", "g_tr", before_objs=[(0, [10, 10, 50, 50])],
                    after_objs=[(0, [20, 20, 60, 60])])
    p_val = _pair("p_va", "g_va", before_objs=[(1, [10, 10, 50, 50])],
                  after_objs=[(1, [20, 20, 60, 60])])
    p_test = _pair("p_te", "g_te", before_objs=[(2, [10, 10, 50, 50])],
                   after_objs=[(2, [20, 20, 60, 60])])
    manifest = _manifest({
        "train": ["p_tr"], "val": ["p_va"], "test": ["p_te"],
    })
    export({"p_tr": p_train, "p_va": p_val, "p_te": p_test}, manifest, tmp_path / "out")

    for split in ("train", "val", "test"):
        d = tmp_path / "out" / "labels" / split
        assert d.exists()
        files = list(d.iterdir())
        assert len(files) == 2, f"{split} should have 2 labels, got {files}"


# ---------- summary counters ----------


def test_export_summary_counts_match_files(tmp_path: Path) -> None:
    p1 = _pair("p1", "g1", before_objs=[(0, [10, 10, 50, 50])],
               after_objs=[(1, [20, 20, 60, 60])])
    p2 = _pair("p2", "g2", before_objs=[(2, [30, 30, 70, 70])],
               after_objs=[(3, [40, 40, 80, 80])])
    manifest = _manifest({"train": ["p1"], "val": ["p2"], "test": []})
    summary = export({"p1": p1, "p2": p2}, manifest, tmp_path / "out")

    assert summary.pairs_processed == 2
    assert summary.images_written == 4
    assert summary.labels_written == 4
    assert summary.empty_labels_written == 0


# ---------- idempotence ----------


def test_export_overwrites_same_stem_on_rerun(tmp_path: Path) -> None:
    pair = _pair("p1", "g1", before_objs=[(0, [10, 10, 50, 50])],
                 after_objs=[(0, [20, 20, 60, 60])])
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "out"
    export({"p1": pair}, manifest, out)
    first = (out / "labels" / "train" / "p1_b.txt").read_text(encoding="utf-8")
    export({"p1": pair}, manifest, out)
    second = (out / "labels" / "train" / "p1_b.txt").read_text(encoding="utf-8")
    assert first == second


# ---------- class names override ----------


def test_export_respects_custom_class_names(tmp_path: Path) -> None:
    pair = _pair("p1", "g1", before_objs=[(0, [10, 10, 50, 50])],
                 after_objs=[])
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    export({"p1": pair}, manifest, tmp_path / "out",
           class_names=("alpha", "beta", "gamma"))

    yaml_text = (tmp_path / "out" / "dataset.yaml").read_text(encoding="utf-8")
    assert "nc: 3" in yaml_text
    assert "alpha" in yaml_text
    assert DEFAULT_CLASS_NAMES[0] not in yaml_text or "bottle" not in yaml_text


# ---------- output dir created if missing ----------


def test_export_creates_output_dir(tmp_path: Path) -> None:
    pair = _pair("p1", "g1", before_objs=[(0, [10, 10, 50, 50])],
                 after_objs=[])
    manifest = _manifest({"train": ["p1"], "val": [], "test": []})
    out = tmp_path / "nested" / "out"
    assert not out.exists()
    export({"p1": pair}, manifest, out)
    assert out.exists()
    assert (out / "dataset.yaml").exists()
