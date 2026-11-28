"""Tests for the read-only dataset split report. Synthetic fixtures only.

The fixtures never create image files: every `before_image_path` /
`after_image_path` points at a file that does not exist. A report that builds
from them therefore proves no image was opened, decoded, stat-ed or hashed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deskcheck.evaluation.dataset_report import (
    build_dataset_report,
    render_markdown,
    report_to_json,
)

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "report_dataset_splits.py"
RELEASE = "v1"


def _obj(object_id: str, class_id: int) -> dict:
    return {"object_id": object_id, "class_id": class_id, "bbox_xyxy": [1, 2, 15, 18]}


def _row(
    pair_id: str,
    group_id: str,
    events: list[tuple[int, str]],
    *,
    no_change: bool = False,
) -> dict:
    """Build one consistent pair record. `events` is a list of (class_id, status)."""
    before: list[dict] = []
    after: list[dict] = []
    records: list[dict] = []
    correspondences: list[dict] = []
    for index, (class_id, status) in enumerate(events):
        before_id = f"{pair_id}-b{index}"
        after_id = f"{pair_id}-a{index}"
        if status == "added":
            after.append(_obj(after_id, class_id))
            before_ref, after_ref = None, after_id
        elif status == "removed":
            before.append(_obj(before_id, class_id))
            before_ref, after_ref = before_id, None
        else:
            before.append(_obj(before_id, class_id))
            after.append(_obj(after_id, class_id))
            before_ref, after_ref = before_id, after_id
            correspondences.append({
                "before_object_id": before_id,
                "after_object_id": after_id,
                "class_id": class_id,
            })
        records.append({
            "class_id": class_id,
            "status": status,
            "before_object_id": before_ref,
            "after_object_id": after_ref,
        })
    return _manual_row(
        pair_id,
        group_id,
        before=before,
        after=after,
        events=records,
        no_change=no_change,
        correspondences=correspondences,
    )


def _manual_row(
    pair_id: str,
    group_id: str | None,
    *,
    before: list[dict],
    after: list[dict],
    events: list[dict],
    no_change: bool = False,
    correspondences: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "pair_id": pair_id,
        "capture_group_id": group_id,
        "before_image_id": f"{pair_id}-before",
        "after_image_id": f"{pair_id}-after",
        "before_image_path": f"{pair_id}-before.png",
        "after_image_path": f"{pair_id}-after.png",
        "width": 64,
        "height": 48,
        "rules_version": "fixture",
        "independently_captured_no_change": no_change,
        "manual_before_objects": before,
        "manual_after_objects": after,
        "correspondences": correspondences or [],
        "events": events,
    }


def _clean_rows() -> list[dict]:
    return [
        _row("p_tr1", "g_tr1", [(0, "unchanged")], no_change=True),
        _row("p_tr2", "g_tr2", [(1, "moved")]),
        _row("p_tr3", "g_tr3", [(2, "added")]),
        _row("p_tr4", "g_tr4", [(3, "removed")]),
        _row("p_va1", "g_va1", [(4, "added")]),
        _row("p_va2", "g_va2", [(5, "removed")]),
        _row("p_va3", "g_va3", [(1, "moved"), (2, "unchanged")]),
        _row("p_va4", "g_va4", [(3, "unchanged")], no_change=True),
        _row("p_te1", "g_te1", [(5, "added")]),
        _row("p_te2", "g_te2", [(4, "removed")]),
        _row("p_te3", "g_te3", [(0, "moved"), (1, "unchanged")]),
        _row("p_te4", "g_te4", [(2, "unchanged")], no_change=True),
    ]


def _clean_splits() -> dict[str, list[str]]:
    return {
        "train": ["p_tr1", "p_tr2", "p_tr3", "p_tr4"],
        "val": ["p_va1", "p_va2", "p_va3", "p_va4"],
        "test": ["p_te1", "p_te2", "p_te3", "p_te4"],
    }


def _write_dataset(
    tmp_path: Path,
    rows: list[dict],
    splits: dict[str, list[str]],
    *,
    release: str = RELEASE,
    directory: str = "dataset",
) -> tuple[Path, Path]:
    base = Path(tmp_path) / directory
    base.mkdir(parents=True)
    pairs_path = base / "pairs.jsonl"
    pairs_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    splits_path = base / "splits.json"
    splits_path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "seed": 7,
            "dataset_release": release,
            "splits": splits,
        }),
        encoding="utf-8",
    )
    return pairs_path, splits_path


def _report(tmp_path: Path, rows: list[dict], splits: dict[str, list[str]]) -> dict:
    pairs_path, splits_path = _write_dataset(tmp_path, rows, splits)
    return build_dataset_report(pairs_path, splits_path, RELEASE)


def _check(report: dict, check_id: str) -> dict:
    matches = [check for check in report["checks"] if check["id"] == check_id]
    assert len(matches) == 1, f"expected exactly one {check_id!r} check"
    return matches[0]


def _run_cli(
    pairs: Path,
    splits: Path,
    output_dir: Path,
    *extra: str,
    release: str = RELEASE,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--pairs", str(pairs),
            "--splits", str(splits),
            "--dataset-release", release,
            "--output-dir", str(output_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=ROOT,
    )


def test_clean_dataset_passes_every_check_and_counts_are_exact(tmp_path: Path) -> None:
    report = _report(tmp_path, _clean_rows(), _clean_splits())

    assert report["summary"] == {
        "check_count": 11,
        "passed": 11,
        "failed": 0,
        "unverified": 0,
        "ok": True,
    }
    assert [check["status"] for check in report["checks"]] == ["pass"] * 11

    train = report["splits"]["train"]
    assert (train["pair_count"], train["image_count"]) == (4, 8)
    assert train["capture_group_count"] == 4
    assert train["object_count"] == 6
    assert train["objects_by_class"] == {
        "bottle": 2, "cup": 2, "keyboard": 1, "mouse": 1, "book": 0, "scissors": 0,
    }
    assert train["events_by_status"] == {
        "added": 1, "moved": 1, "removed": 1, "unchanged": 1,
    }
    assert train["pairs_by_status"] == {
        "added": 1, "moved": 1, "removed": 1, "unchanged": 1,
    }
    assert (train["no_change_pairs"], train["eligible_no_change_pairs"]) == (1, 1)

    val = report["splits"]["val"]
    assert (val["pair_count"], val["image_count"], val["object_count"]) == (4, 8, 8)
    assert val["events_by_status"]["unchanged"] == 2
    assert val["pairs_by_status"]["unchanged"] == 2

    totals = report["splits"]["total"]
    assert (totals["pair_count"], totals["image_count"]) == (12, 24)
    assert totals["capture_group_count"] == 12
    assert totals["object_count"] == 22
    assert totals["events_by_status"] == {
        "added": 3, "moved": 3, "removed": 3, "unchanged": 5,
    }
    assert (totals["no_change_pairs"], totals["eligible_no_change_pairs"]) == (3, 3)

    assert report["dataset"]["object_count"] == 22
    assert report["dataset"]["pair_count"] == 12
    assert _check(report, "object_counts_reconcile")["detail"] == (
        "objects across splits: 22; objects in the pairs file: 22"
    )


def test_report_records_digests_release_and_metadata_only_statement(tmp_path: Path) -> None:
    rows, splits = _clean_rows(), _clean_splits()
    pairs_path, splits_path = _write_dataset(tmp_path, rows, splits)

    report = build_dataset_report(pairs_path, splits_path, RELEASE)

    assert report["annotation_metadata_only"] is True
    assert "no image file was opened" in report["metadata_only_statement"]
    assert report["dataset_release"] == RELEASE
    assert report["splits_manifest"]["dataset_release"] == RELEASE
    assert report["inputs"]["pairs"]["sha256"] == hashlib.sha256(
        pairs_path.read_bytes()
    ).hexdigest()
    assert report["inputs"]["splits"]["sha256"] == hashlib.sha256(
        splits_path.read_bytes()
    ).hexdigest()
    assert report["inputs"]["pairs"]["name"] == "pairs.jsonl"
    assert json.loads(json.dumps(report)) == report


def test_capture_group_shared_between_train_and_test_is_flagged(tmp_path: Path) -> None:
    rows = _clean_rows()
    for row in rows:
        if row["pair_id"] == "p_te4":
            row["capture_group_id"] = "g_tr4"

    report = _report(tmp_path, rows, _clean_splits())

    leakage = _check(report, "capture_group_leakage")
    assert leakage["status"] == "fail"
    assert leakage["offending"] == [
        {"capture_group_id": "g_tr4", "splits": ["train", "test"]},
    ]
    assert "1 of 11 capture group(s)" in leakage["detail"]
    assert report["summary"]["ok"] is False
    assert "g_tr4" in render_markdown(report)


def test_pair_absent_from_splits_is_flagged_with_its_id(tmp_path: Path) -> None:
    splits = _clean_splits()
    splits["train"] = splits["train"][:-1]

    report = _report(tmp_path, _clean_rows(), splits)

    assigned = _check(report, "pairs_file_pair_ids_assigned")
    assert assigned["status"] == "fail"
    assert assigned["offending"] == ["p_tr4"]
    reconcile = _check(report, "object_counts_reconcile")
    assert reconcile["status"] == "fail"
    assert reconcile["offending"] == [{"across_splits": 21, "pairs_file": 22}]


def test_status_missing_from_val_is_named(tmp_path: Path) -> None:
    rows = [row for row in _clean_rows() if row["pair_id"] != "p_va1"]
    splits = _clean_splits()
    splits["val"] = ["p_va2", "p_va3", "p_va4"]

    report = _report(tmp_path, rows, splits)

    val = _check(report, "status_coverage_val")
    assert val["status"] == "fail"
    assert val["offending"] == ["added"]
    assert val["detail"] == "missing from val: ['added']"
    assert _check(report, "status_coverage_test")["status"] == "pass"


@pytest.mark.parametrize("bad_group", [None, "", "   "])
def test_unusable_capture_group_field_is_cannot_be_verified(
    tmp_path: Path, bad_group: object
) -> None:
    rows = _clean_rows()
    for row in rows:
        if row["pair_id"] == "p_te4":
            row["capture_group_id"] = bad_group

    report = _report(tmp_path, rows, _clean_splits())

    leakage = _check(report, "capture_group_leakage")
    assert leakage["status"] == "cannot_be_verified"
    assert leakage["status"] != "pass"
    assert "capture_group_id" in leakage["detail"]
    assert leakage["offending"] == ["p_te4"]
    assert report["summary"]["unverified"] == 1
    assert report["summary"]["ok"] is False


def test_absent_capture_group_field_is_rejected_by_the_loader(tmp_path: Path) -> None:
    rows = _clean_rows()
    del rows[0]["capture_group_id"]
    pairs_path, splits_path = _write_dataset(tmp_path, rows, _clean_splits())

    with pytest.raises(ValueError) as raised:
        build_dataset_report(pairs_path, splits_path, RELEASE)

    assert str(raised.value) == "pair on line 1: missing fields ['capture_group_id']"


def test_release_mismatch_is_rejected(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(
        tmp_path, _clean_rows(), _clean_splits(), release="v2"
    )

    with pytest.raises(ValueError, match="dataset release mismatch"):
        build_dataset_report(pairs_path, splits_path, "v1")


def test_pair_in_two_splits_is_rejected_by_the_loader(tmp_path: Path) -> None:
    splits = _clean_splits()
    splits["val"].append("p_tr1")
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), splits)

    with pytest.raises(ValueError, match="appears in both"):
        build_dataset_report(pairs_path, splits_path, RELEASE)


def test_unknown_class_id_is_flagged(tmp_path: Path) -> None:
    rows = _clean_rows()
    rows[1] = _row("p_tr2", "g_tr2", [(9, "moved")])

    report = _report(tmp_path, rows, _clean_splits())

    known = _check(report, "class_ids_known")
    assert known["status"] == "fail"
    assert known["offending"] == [
        {
            "pair_id": "p_tr2",
            "side": "manual_before_objects",
            "object_id": "p_tr2-b0",
            "class_id": 9,
        },
        {
            "pair_id": "p_tr2",
            "side": "manual_after_objects",
            "object_id": "p_tr2-a0",
            "class_id": 9,
        },
    ]
    assert report["splits"]["train"]["other_class_object_count"] == 2
    assert "(other class ids)" in render_markdown(report)


def test_status_and_box_presence_disagreement_is_flagged(tmp_path: Path) -> None:
    bad = _manual_row(
        "p_tr2",
        "g_tr2",
        before=[_obj("p_tr2-b0", 1)],
        after=[_obj("p_tr2-a0", 1)],
        events=[{
            "class_id": 1,
            "status": "added",
            "before_object_id": "p_tr2-b0",
            "after_object_id": "p_tr2-a0",
        }],
    )
    rows = _clean_rows()
    rows[1] = bad

    report = _report(tmp_path, rows, _clean_splits())

    presence = _check(report, "status_box_presence_agrees")
    assert presence["status"] == "fail"
    assert presence["offending"] == [{
        "pair_id": "p_tr2",
        "class_id": 1,
        "status": "added",
        "before_box_present": True,
        "after_box_present": True,
    }]
    assert _check(report, "annotation_converter_accepts_pairs")["status"] == "pass"


def test_eligible_no_change_pair_with_change_events_is_flagged(tmp_path: Path) -> None:
    rows = _clean_rows()
    rows[0] = _row("p_tr1", "g_tr1", [(0, "moved")], no_change=True)

    report = _report(tmp_path, rows, _clean_splits())

    converter = _check(report, "annotation_converter_accepts_pairs")
    assert converter["status"] == "fail"
    assert converter["offending"][0]["pair_id"] == "p_tr1"
    assert "no-change" in converter["offending"][0]["error"]


def test_report_does_not_stat_image_paths(tmp_path: Path, monkeypatch) -> None:
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), _clean_splits())
    statted: list[str] = []
    real_stat = Path.stat

    def _recording_stat(self: Path, *args, **kwargs):
        statted.append(str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _recording_stat)
    report = build_dataset_report(pairs_path, splits_path, RELEASE)

    assert report["summary"]["ok"] is True
    assert [path for path in statted if path.endswith(".png")] == []
    assert list(tmp_path.rglob("*.png")) == []


def test_report_json_is_deterministic_and_sorted(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), _clean_splits())

    first = report_to_json(build_dataset_report(pairs_path, splits_path, RELEASE))
    second = report_to_json(build_dataset_report(pairs_path, splits_path, RELEASE))

    assert first == second
    payload = json.loads(first)
    assert first == json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert list(payload) == sorted(payload)


def test_cli_writes_report_json_and_markdown(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), _clean_splits())
    output = tmp_path / "out" / "report"

    completed = _run_cli(pairs_path, splits_path, output)

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "failed=0 unverified=0" in completed.stdout
    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert payload["summary"]["ok"] is True
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Dataset split report")
    assert "## Integrity checks" in markdown
    assert "None; every check passed." in markdown


def test_cli_output_is_byte_identical_for_identical_input(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), _clean_splits())
    first, second = tmp_path / "out" / "first", tmp_path / "out" / "second"

    assert _run_cli(pairs_path, splits_path, first).returncode == 0
    assert _run_cli(pairs_path, splits_path, second).returncode == 0

    for name in ("report.json", "report.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_cli_refuses_existing_output_directory(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), _clean_splits())
    output = tmp_path / "report"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    completed = _run_cli(pairs_path, splits_path, output)

    assert completed.returncode == 2
    assert "refusing" in completed.stderr
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_cli_failed_run_leaves_no_output(tmp_path: Path) -> None:
    splits = _clean_splits()
    splits["val"].append("p_missing")
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), splits)
    output = tmp_path / "out" / "report"

    completed = _run_cli(pairs_path, splits_path, output)

    assert completed.returncode == 2
    assert "manifest references unknown pair_ids: ['p_missing']" in completed.stderr
    assert not output.exists()
    assert not output.parent.exists()
    assert list(tmp_path.rglob(".report-*")) == []
    assert list(tmp_path.rglob("report.json")) == []


def test_cli_release_mismatch_writes_nothing(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(
        tmp_path, _clean_rows(), _clean_splits(), release="v2"
    )
    output = tmp_path / "out" / "report"

    completed = _run_cli(pairs_path, splits_path, output, release="v1")

    assert completed.returncode == 2
    assert "dataset release mismatch" in completed.stderr
    assert not output.exists()


def test_cli_expectations_add_a_separate_markdown_section(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), _clean_splits())
    output = tmp_path / "report"

    completed = _run_cli(
        pairs_path, splits_path, output,
        "--expect-pairs", "14,4,6", "--expect-objects", "269",
    )

    assert completed.returncode == 0, completed.stderr
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "## Differences from expected" in markdown
    assert "| pairs (train) | 14 | 4 | differs |" in markdown
    assert "| pairs (val) | 4 | 4 | match |" in markdown
    assert "| objects (pairs file) | 269 | 22 | differs |" in markdown
    payload = (output / "report.json").read_text(encoding="utf-8")
    assert "269" not in payload
    assert "expected" not in payload
    assert json.loads(payload)["summary"]["ok"] is True


def test_markdown_without_expectations_has_no_difference_section(tmp_path: Path) -> None:
    markdown = render_markdown(_report(tmp_path, _clean_rows(), _clean_splits()))

    assert "Differences from expected" not in markdown
    assert "expected" not in markdown


def test_cli_rejects_malformed_expect_pairs(tmp_path: Path) -> None:
    pairs_path, splits_path = _write_dataset(tmp_path, _clean_rows(), _clean_splits())
    output = tmp_path / "report"

    completed = _run_cli(pairs_path, splits_path, output, "--expect-pairs", "14,4")

    assert completed.returncode == 2
    assert "expected three comma-separated counts" in completed.stderr
    assert not output.exists()
