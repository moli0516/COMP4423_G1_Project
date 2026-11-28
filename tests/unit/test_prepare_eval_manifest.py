"""Tests for the evaluation manifest builder, verified through the real load_manifest."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deskcheck.evaluation.eval_manifest import build_eval_manifest, publish_manifest

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "prepare_eval_manifest.py"
IMAGE_BYTES = b"fixture-bytes-not-a-decodable-image"
RELEASE = "fixture-release"


def _runner_load_manifest():
    spec = importlib.util.spec_from_file_location(
        "eval_manifest_runner", ROOT / "scripts" / "evaluate_real_pairs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_manifest


load_manifest = _runner_load_manifest()


def _obj(object_id: str, class_id: int = 0) -> dict:
    return {"object_id": object_id, "class_id": class_id, "bbox_xyxy": [1, 2, 15, 18]}


def _event(class_id: int, status: str, before: str | None, after: str | None) -> dict:
    return {
        "class_id": class_id,
        "status": status,
        "before_object_id": before,
        "after_object_id": after,
    }


def _row(
    pair_id: str,
    *,
    events: list[dict] | None = None,
    before: list[dict] | None = None,
    after: list[dict] | None = None,
    no_change: bool = False,
) -> dict:
    before = before if before is not None else [_obj(f"{pair_id}-b")]
    after = after if after is not None else [_obj(f"{pair_id}-a")]
    events = events if events is not None else [
        _event(0, "unchanged", f"{pair_id}-b", f"{pair_id}-a")
    ]
    return {
        "schema_version": "1.0",
        "pair_id": pair_id,
        "capture_group_id": pair_id,
        "width": 64,
        "height": 48,
        "rules_version": "fixture",
        "before_image_id": f"{pair_id}-before",
        "after_image_id": f"{pair_id}-after",
        "independently_captured_no_change": no_change,
        "manual_before_objects": before,
        "manual_after_objects": after,
        "correspondences": [],
        "events": events,
    }


def _rows() -> list[dict]:
    return [_row("p1"), _row("p2", no_change=True)]


def _photos(pairs_path: Path) -> Path:
    return pairs_path.parent.parent / "photos"


def _make_release(
    tmp_path: Path,
    rows: list[dict],
    *,
    splits: dict[str, list[str]] | None = None,
    release: str = RELEASE,
) -> tuple[Path, Path]:
    """Write synthetic pairs.jsonl, splits.json and byte-only photos (never decoded)."""
    annotations = tmp_path / "release" / "annotations" / "pairs.jsonl"
    photos = _photos(annotations)
    photos.mkdir(parents=True)
    annotations.parent.mkdir(parents=True)
    lines = []
    for row in rows:
        for side in ("before", "after"):
            name = f"{row['pair_id']}_{side}.png"
            (photos / name).write_bytes(IMAGE_BYTES)
            row[f"{side}_image_path"] = f"../photos/{name}"
        lines.append(json.dumps(row))
    annotations.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    if splits is None:
        splits = {"train": [], "val": [row["pair_id"] for row in rows], "test": []}
    splits_path = annotations.parent / "splits.json"
    splits_path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "seed": 42,
            "dataset_release": release,
            "splits": splits,
        }),
        encoding="utf-8",
    )
    return annotations, splits_path


def _three_split_release(tmp_path: Path) -> tuple[Path, Path]:
    rows = [_row("tr1"), _row("tr2"), _row("v1"), _row("te1"), _row("te2")]
    splits = {"train": ["tr1", "tr2"], "val": ["v1"], "test": ["te1", "te2"]}
    return _make_release(tmp_path, rows, splits=splits)


def _build(
    pairs: Path,
    splits: Path,
    output: Path,
    *,
    split: str = "val",
    release: str = RELEASE,
) -> dict:
    return build_eval_manifest(
        pairs, splits_path=splits, split=split, dataset_release=release, output=output
    )


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


def test_writes_manifest_the_runner_accepts(tmp_path: Path) -> None:
    pairs, splits = _make_release(tmp_path, _rows())
    output = tmp_path / "out" / "reports" / "manifest.json"

    publish_manifest(_build(pairs, splits, output), output, verify=load_manifest)

    loaded = load_manifest(output)
    assert loaded["split"] == "val"
    assert loaded["dataset_release"] == RELEASE
    assert [row["pair_id"] for row in loaded["pairs"]] == ["p1", "p2"]
    assert loaded["eligible_no_change_pair_ids"] == ["p2"]
    assert loaded["ground_truth"]["p1"] == [
        {
            "pair_id": "p1",
            "class_id": 0,
            "status": "unchanged",
            "before_box": [1, 2, 15, 18],
            "after_box": [1, 2, 15, 18],
        }
    ]
    photos = _photos(pairs)
    for row in loaded["pairs"]:
        assert row["before_path"] == photos / f"{row['pair_id']}_before.png"
        assert row["after_path"] == photos / f"{row['pair_id']}_after.png"


def test_only_selected_split_pairs_are_emitted(tmp_path: Path) -> None:
    pairs, splits = _three_split_release(tmp_path)

    val = _build(pairs, splits, tmp_path / "out" / "val.json", split="val")
    test = _build(pairs, splits, tmp_path / "out" / "test.json", split="test")

    assert [row["pair_id"] for row in val["pairs"]] == ["v1"]
    assert list(val["ground_truth"]) == ["v1"]
    assert [row["pair_id"] for row in test["pairs"]] == ["te1", "te2"]
    assert list(test["ground_truth"]) == ["te1", "te2"]
    assert test["split"] == "test"


def test_other_split_defects_are_not_inspected(tmp_path: Path) -> None:
    broken_class = _row(
        "tr1",
        before=[_obj("tr1-b", 9)],
        after=[_obj("tr1-a", 9)],
        events=[_event(9, "unchanged", "tr1-b", "tr1-a")],
    )
    rows = [broken_class, _row("tr2"), _row("te1")]
    pairs, splits = _make_release(
        tmp_path, rows, splits={"train": ["tr1", "tr2"], "val": [], "test": ["te1"]}
    )
    (_photos(pairs) / "tr2_after.png").unlink()

    output = tmp_path / "out" / "test.json"
    publish_manifest(_build(pairs, splits, output, split="test"), output, verify=load_manifest)

    loaded = load_manifest(output)
    assert [row["pair_id"] for row in loaded["pairs"]] == ["te1"]
    assert set(loaded["ground_truth"]) == {"te1"}
    assert not (_photos(pairs) / "tr2_after.png").exists()


def test_splits_listing_absent_pair_id_is_rejected(tmp_path: Path) -> None:
    pairs, splits = _make_release(
        tmp_path,
        [_row("te1")],
        splits={"train": [], "val": [], "test": ["te1", "ghost"]},
    )

    with pytest.raises(ValueError, match="ghost"):
        _build(pairs, splits, tmp_path / "out" / "test.json", split="test")


def test_pair_in_two_splits_is_rejected(tmp_path: Path) -> None:
    pairs, splits = _make_release(
        tmp_path,
        [_row("te1")],
        splits={"train": [], "val": ["te1"], "test": ["te1"]},
    )

    with pytest.raises(ValueError, match="appears in both"):
        _build(pairs, splits, tmp_path / "out" / "test.json", split="test")


def test_empty_selected_split_is_rejected(tmp_path: Path) -> None:
    pairs, splits = _make_release(
        tmp_path,
        [_row("te1")],
        splits={"train": [], "val": [], "test": ["te1"]},
    )

    with pytest.raises(ValueError, match=r"split 'val' selects no pairs"):
        _build(pairs, splits, tmp_path / "out" / "val.json", split="val")


def test_dataset_release_mismatch_is_rejected(tmp_path: Path) -> None:
    pairs, splits = _make_release(tmp_path, _rows(), release="released-v1")

    with pytest.raises(ValueError, match="dataset release mismatch"):
        _build(pairs, splits, tmp_path / "out" / "manifest.json", release="other-v9")


def test_image_paths_are_relative_to_the_manifest_file(tmp_path: Path) -> None:
    pairs, splits = _make_release(tmp_path, _rows())
    output = tmp_path / "out" / "reports" / "nested" / "manifest.json"

    publish_manifest(_build(pairs, splits, output), output, verify=load_manifest)

    text = output.read_text(encoding="utf-8")
    written = json.loads(text)
    assert str(tmp_path) not in text
    assert not any(Path(row[key]).is_absolute()
                   for row in written["pairs"] for key in ("before_path", "after_path"))
    assert all(".." in row["before_path"] for row in written["pairs"])
    loaded = load_manifest(output)
    for row in loaded["pairs"]:
        assert row["before_path"].is_file()


def test_same_input_gives_identical_bytes(tmp_path: Path) -> None:
    pairs, splits = _make_release(tmp_path, _rows())
    first = tmp_path / "out" / "a.json"
    second = tmp_path / "out" / "b.json"

    publish_manifest(_build(pairs, splits, first), first, verify=load_manifest)
    publish_manifest(_build(pairs, splits, second), second, verify=load_manifest)

    assert first.read_bytes() == second.read_bytes()


def test_empty_pairs_file_is_rejected(tmp_path: Path) -> None:
    pairs, splits = _make_release(
        tmp_path, [], splits={"train": [], "val": [], "test": []}
    )

    with pytest.raises(ValueError, match="no pairs"):
        _build(pairs, splits, tmp_path / "out" / "manifest.json")


def test_duplicate_pair_ids_are_rejected(tmp_path: Path) -> None:
    pairs, splits = _make_release(
        tmp_path, [_row("p1"), _row("p1")], splits={"train": [], "val": ["p1"], "test": []}
    )

    with pytest.raises(ValueError, match="duplicate pair_id 'p1'"):
        _build(pairs, splits, tmp_path / "out" / "manifest.json")


def test_status_and_box_presence_disagreement_is_rejected(tmp_path: Path) -> None:
    # "added" must have no before box, but both object references resolve.
    bad = _row("p1", events=[_event(0, "added", "p1-b", "p1-a")])
    pairs, splits = _make_release(tmp_path, [bad])

    with pytest.raises(ValueError, match=r"pair 'p1': status 'added'.*box presence"):
        _build(pairs, splits, tmp_path / "out" / "manifest.json")


def test_unknown_class_is_rejected(tmp_path: Path) -> None:
    bad = _row(
        "p1",
        before=[_obj("p1-b", 6)],
        after=[_obj("p1-a", 6)],
        events=[_event(6, "unchanged", "p1-b", "p1-a")],
    )
    pairs, splits = _make_release(tmp_path, [bad])

    with pytest.raises(ValueError, match=r"pair 'p1': unknown class_id 6"):
        _build(pairs, splits, tmp_path / "out" / "manifest.json")


def test_eligible_no_change_pair_with_change_events_is_rejected(tmp_path: Path) -> None:
    bad = _row("p2", no_change=True, events=[_event(0, "moved", "p2-b", "p2-a")])
    pairs, splits = _make_release(tmp_path, [_row("p1"), bad])

    with pytest.raises(ValueError, match=r"pair 'p2': eligible no-change pair has change"):
        _build(pairs, splits, tmp_path / "out" / "manifest.json")


def test_missing_image_file_is_rejected_naming_pair_and_side(tmp_path: Path) -> None:
    pairs, splits = _make_release(tmp_path, _rows())
    (_photos(pairs) / "p1_after.png").unlink()

    with pytest.raises(FileNotFoundError, match=r"pair 'p1': after_path does not exist"):
        _build(pairs, splits, tmp_path / "out" / "manifest.json")


def test_existing_output_is_refused_and_untouched(tmp_path: Path) -> None:
    pairs, splits = _make_release(tmp_path, _rows())
    output = tmp_path / "out" / "manifest.json"
    output.parent.mkdir()
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_manifest(_build(pairs, splits, output), output, verify=load_manifest)

    assert output.read_bytes() == b"keep"
    assert _leftovers(output.parent) == []


def test_failed_verification_publishes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "out" / "manifest.json"
    output.parent.mkdir()

    # The real verifier rejects payloads that are not complete manifests.
    with pytest.raises(ValueError):
        publish_manifest({"split": "val"}, output, verify=load_manifest)

    assert not output.exists()
    assert _leftovers(output.parent) == []


def _run_cli(
    pairs: Path,
    splits: Path,
    output: Path,
    *,
    split: str = "val",
    release: str = RELEASE,
):
    return subprocess.run(
        [sys.executable, str(CLI), "--pairs", str(pairs), "--splits", str(splits),
         "--split", split, "--dataset-release", release, "--output", str(output)],
        capture_output=True, text=True, check=False, timeout=60, cwd=ROOT,
    )


def test_cli_writes_val_and_test_manifests(tmp_path: Path) -> None:
    pairs, splits = _three_split_release(tmp_path)

    for split, expected in (("val", ["v1"]), ("test", ["te1", "te2"])):
        output = tmp_path / "out" / f"manifest_{split}.json"
        completed = _run_cli(pairs, splits, output, split=split)
        assert completed.returncode == 0, completed.stderr
        assert f"split={split} pairs={len(expected)}" in completed.stdout
        loaded = load_manifest(output)
        assert loaded["split"] == split
        assert [row["pair_id"] for row in loaded["pairs"]] == expected


def test_cli_rejects_train_split(tmp_path: Path) -> None:
    pairs, splits = _three_split_release(tmp_path)
    output = tmp_path / "out" / "manifest_train.json"

    completed = _run_cli(pairs, splits, output, split="train")

    assert completed.returncode == 2
    assert not output.exists()


def test_cli_rejection_leaves_no_manifest(tmp_path: Path) -> None:
    bad = _row(
        "p1",
        before=[_obj("p1-b", 9)],
        after=[_obj("p1-a", 9)],
        events=[_event(9, "unchanged", "p1-b", "p1-a")],
    )
    pairs, splits = _make_release(tmp_path, [bad])
    output = tmp_path / "out" / "manifest.json"

    completed = _run_cli(pairs, splits, output)

    assert completed.returncode == 2
    assert "unknown class_id 9" in completed.stderr
    assert not output.exists()
    assert not output.parent.exists()


def test_cli_refuses_existing_output(tmp_path: Path) -> None:
    pairs, splits = _make_release(tmp_path, _rows())
    output = tmp_path / "out" / "manifest.json"
    output.parent.mkdir()
    output.write_bytes(b"keep")

    completed = _run_cli(pairs, splits, output)

    assert completed.returncode == 2
    assert "refusing" in completed.stderr
    assert output.read_bytes() == b"keep"
