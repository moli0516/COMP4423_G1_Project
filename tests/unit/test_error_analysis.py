"""Tests for read-only error analysis, using the real evaluators and load_manifest."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deskcheck.evaluation import error_analysis
from deskcheck.evaluation.error_analysis import analyze_errors
from deskcheck.evaluation.event_evaluator import EventEvaluator

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "analyze_errors.py"
BOX = [0.0, 0.0, 10.0, 10.0]
FAR = [100.0, 100.0, 110.0, 110.0]
SHIFTED = [2.0, 0.0, 12.0, 10.0]  # IoU 2/3 with BOX, between 0.5 and 0.9
IOU = 0.5
STATUSES = ("added", "removed", "moved", "unchanged")


def _runner_load_manifest():
    spec = importlib.util.spec_from_file_location(
        "error_analysis_runner", ROOT / "scripts" / "evaluate_real_pairs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_manifest


load_manifest = _runner_load_manifest()


def _event(class_id: int, status: str, box: list[float] = BOX) -> dict:
    before = None if status == "added" else list(box)
    after = None if status == "removed" else list(box)
    return {
        "class_id": class_id,
        "status": status,
        "before_box": before,
        "after_box": after,
    }


def _write_manifest(
    tmp_path: Path,
    ground_truth: dict,
    *,
    eligible: tuple[str, ...] = (),
    split: str = "val",
    release: str = "fixture-v1",
) -> Path:
    photos = tmp_path / "photos"
    photos.mkdir(exist_ok=True)
    pairs = []
    for pair_id in sorted(ground_truth):
        for side in ("before", "after"):
            (photos / f"{pair_id}_{side}.png").write_bytes(b"fixture-bytes-not-an-image")
        pairs.append({
            "pair_id": pair_id,
            "before_path": f"photos/{pair_id}_before.png",
            "after_path": f"photos/{pair_id}_after.png",
        })
    manifest = {
        "schema_version": "1.0",
        "dataset_release": release,
        "split": split,
        "pairs": pairs,
        "ground_truth": ground_truth,
        "eligible_no_change_pair_ids": list(eligible),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_predictions(tmp_path: Path, predictions: dict, name: str = "predictions.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(predictions), encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    ground_truth: dict,
    predictions: dict,
    *,
    eligible: tuple[str, ...] = (),
    cases_per_group: int = 2,
    iou_threshold: float = IOU,
    loader=load_manifest,
    out_name: str = "analysis",
    manifest_path: Path | None = None,
) -> Path:
    manifest = manifest_path or _write_manifest(tmp_path, ground_truth, eligible=eligible)
    preds = _write_predictions(tmp_path, predictions)
    output = tmp_path / out_name
    analyze_errors(
        preds,
        manifest,
        output,
        iou_threshold=iou_threshold,
        manifest_loader=loader,
        cases_per_group=cases_per_group,
    )
    return output


def _csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pair_rows(output: Path) -> dict[str, dict]:
    return {row["pair_id"]: row for row in _csv_rows(output / "pair_results.csv")}


def _event_rows(output: Path) -> list[dict]:
    return _csv_rows(output / "event_errors.csv")


def _cases(output: Path) -> dict:
    return json.loads((output / "selected_cases.json").read_text(encoding="utf-8"))


def _summary(output: Path) -> dict:
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("status", STATUSES)
def test_correct_event_for_each_status(tmp_path: Path, status: str) -> None:
    ground_truth = {"p1": [_event(0, status)]}
    predictions = {"p1": [_event(0, status)]}

    output = _run(tmp_path, ground_truth, predictions)

    row = _pair_rows(output)["p1"]
    assert row["tp"] == "1" and row["fn"] == "0" and row["fp"] == "0"
    assert row["all_correct"] == "true"
    assert row["gt_event_count"] == row["pred_event_count"] == "1"
    assert _event_rows(output) == []


@pytest.mark.parametrize(
    ("gt_status", "pred_status"),
    [("added", "removed"), ("removed", "added"), ("moved", "unchanged"),
     ("unchanged", "moved")],
)
def test_wrong_status_for_each_status_pair(
    tmp_path: Path, gt_status: str, pred_status: str
) -> None:
    ground_truth = {"p1": [_event(0, gt_status)]}
    predictions = {"p1": [_event(0, pred_status)]}

    output = _run(tmp_path, ground_truth, predictions)

    row = _pair_rows(output)["p1"]
    assert row["tp"] == "0" and row["fn"] == "1" and row["fp"] == "1"
    assert row["all_correct"] == "false"
    (error,) = _event_rows(output)
    assert error["error_type"] == "wrong_status"
    assert error["gt_status"] == gt_status
    assert error["pred_status"] == pred_status


def test_missed_event(tmp_path: Path) -> None:
    output = _run(tmp_path, {"p1": [_event(0, "moved")]}, {"p1": []})

    row = _pair_rows(output)["p1"]
    assert row["gt_event_count"] == "1" and row["pred_event_count"] == "0"
    assert row["tp"] == "0" and row["fn"] == "1" and row["fp"] == "0"
    (error,) = _event_rows(output)
    assert error["error_type"] == "missed_event"
    assert error["gt_status"] == "moved"
    assert error["pred_status"] == ""


def test_spurious_event(tmp_path: Path) -> None:
    output = _run(tmp_path, {"p1": []}, {"p1": [_event(1, "added")]})

    row = _pair_rows(output)["p1"]
    assert row["gt_event_count"] == "0" and row["pred_event_count"] == "1"
    assert row["tp"] == "0" and row["fn"] == "0" and row["fp"] == "1"
    (error,) = _event_rows(output)
    assert error["error_type"] == "spurious_event"
    assert error["gt_status"] == ""
    assert error["pred_status"] == "added"


def test_misaligned_event(tmp_path: Path) -> None:
    ground_truth = {"p1": [_event(0, "unchanged", BOX)]}
    predictions = {"p1": [_event(0, "unchanged", FAR)]}

    output = _run(tmp_path, ground_truth, predictions)

    row = _pair_rows(output)["p1"]
    assert row["tp"] == "0" and row["fn"] == "1" and row["fp"] == "1"
    (error,) = _event_rows(output)
    assert error["error_type"] == "misaligned_event"
    assert json.loads(error["gt_before_box"]) == BOX
    assert json.loads(error["pred_before_box"]) == FAR


def test_other_bucket_records_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ZeroEvaluator:
        STATUSES = STATUSES

        def __init__(self, iou_threshold: float = 0.5) -> None:
            self.iou_threshold = iou_threshold

        def evaluate(self, predicted: dict, ground_truth: dict) -> dict:
            return {
                "protocol_version": "zeroed-fixture",
                "iou_threshold": self.iou_threshold,
                "zero_division": 0,
                "per_status": {status: {"tp": 0, "fp": 0, "fn": 0} for status in STATUSES},
            }

    monkeypatch.setattr(error_analysis, "EventEvaluator", ZeroEvaluator)
    ground_truth = {"p1": [_event(0, "unchanged")]}
    predictions = {"p1": [_event(0, "unchanged")]}

    output = _run(tmp_path, ground_truth, predictions)

    (error,) = _event_rows(output)
    assert error["error_type"] == "other"
    cases = _cases(output)
    (group,) = cases["groups"]
    assert group["error_type"] == "other"
    assert "unexpected verdict" in group["cases"][0]["error_reason"]
    summary = _summary(output)
    assert summary["error_type_counts"]["other"] == 1
    assert summary["other_reasons"] == [
        {"pair_id": "p1", "class_id": 0, "reason": group["cases"][0]["error_reason"]}
    ]


def test_no_change_success_and_failure(tmp_path: Path) -> None:
    ground_truth = {
        "p1": [_event(0, "unchanged")],
        "p2": [_event(0, "unchanged")],
    }
    predictions = {
        "p1": [_event(0, "unchanged")],
        "p2": [_event(0, "moved")],
    }

    output = _run(tmp_path, ground_truth, predictions, eligible=("p1", "p2"))

    rows = _pair_rows(output)
    assert rows["p1"]["is_eligible_nochange"] == "true"
    assert rows["p2"]["is_eligible_nochange"] == "true"
    no_change = _cases(output)["no_change"]
    assert no_change["evaluated"] is True
    assert [case["pair_id"] for case in no_change["successes"]["cases"]] == ["p1"]
    assert no_change["successes"]["cases"][0]["error_type"] == "none"
    assert [case["pair_id"] for case in no_change["failures"]["cases"]] == ["p2"]
    assert no_change["failures"]["cases"][0]["error_type"] == "false_alarm"
    assert _summary(output)["rules"]["no_change_protocol_version"]


def test_pair_id_mismatch_names_the_pair(tmp_path: Path) -> None:
    ground_truth = {"p1": [], "p2": []}
    manifest = _write_manifest(tmp_path, ground_truth)

    with pytest.raises(ValueError, match=r"missing from predictions.*p2"):
        _run(tmp_path, ground_truth, {"p1": []}, manifest_path=manifest)

    with pytest.raises(ValueError, match=r"absent from manifest.*p3"):
        _run(tmp_path, ground_truth, {"p1": [], "p2": [], "p3": []},
             manifest_path=manifest, out_name="analysis2")


def test_invalid_manifest_is_rejected_by_load_manifest(tmp_path: Path) -> None:
    ground_truth = {"p1": []}
    manifest = _write_manifest(tmp_path, ground_truth)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["split"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing or unexpected fields"):
        _run(tmp_path, ground_truth, {"p1": []}, manifest_path=manifest)
    assert not (tmp_path / "analysis").exists()


def test_existing_output_dir_is_refused_and_untouched(tmp_path: Path) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _run(tmp_path, {"p1": []}, {"p1": []})

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in output.iterdir()) == ["keep.txt"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_failed_run_leaves_no_output_dir(tmp_path: Path) -> None:
    ground_truth = {"p1": [], "p2": []}
    manifest = _write_manifest(tmp_path, ground_truth)

    with pytest.raises(ValueError):
        _run(tmp_path, ground_truth, {"p1": []}, manifest_path=manifest)

    assert not (tmp_path / "analysis").exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_identical_inputs_give_byte_identical_outputs(tmp_path: Path) -> None:
    ground_truth = {"p1": [_event(0, "moved")], "p2": [_event(1, "added")]}
    predictions = {"p1": [_event(0, "unchanged")], "p2": []}
    manifest = _write_manifest(tmp_path, ground_truth, eligible=())

    first = _run(tmp_path, ground_truth, predictions, manifest_path=manifest, out_name="a")
    second = _run(tmp_path, ground_truth, predictions, manifest_path=manifest, out_name="b")

    for name in ("pair_results.csv", "event_errors.csv", "selected_cases.json",
                 "summary.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_selection_limits_and_orders_by_pair_id(tmp_path: Path) -> None:
    ground_truth = {pair_id: [_event(0, "moved")] for pair_id in ("c1", "a1", "b1")}
    predictions = {pair_id: [] for pair_id in ground_truth}

    limited = _run(tmp_path, ground_truth, predictions, cases_per_group=2, out_name="lim")
    (group,) = _cases(limited)["groups"]
    assert group["status"] == "moved" and group["error_type"] == "missed_event"
    assert group["total"] == 3
    assert [case["pair_id"] for case in group["cases"]] == ["a1", "b1"]
    assert [case["pair_id"] for case in group["cases"]] == sorted(
        case["pair_id"] for case in group["cases"]
    )
    for case in group["cases"]:
        assert set(case) == {"pair_id", "class_id", "gt_status", "pred_status",
                            "error_type", "reason", "error_reason"}

    full = _run(tmp_path, ground_truth, predictions, cases_per_group=10, out_name="all")
    (group_all,) = _cases(full)["groups"]
    assert [case["pair_id"] for case in group_all["cases"]] == ["a1", "b1", "c1"]


def test_cases_per_group_must_be_positive(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, {"p1": []})
    preds = _write_predictions(tmp_path, {"p1": []})

    with pytest.raises(ValueError, match="cases_per_group"):
        analyze_errors(preds, manifest, tmp_path / "out", iou_threshold=IOU,
                       manifest_loader=load_manifest, cases_per_group=0)


def test_counts_match_the_evaluators_aggregate(tmp_path: Path) -> None:
    ground_truth = {
        "p1": [_event(0, "added"), _event(1, "removed")],
        "p2": [_event(2, "moved"), _event(3, "unchanged")],
        "p3": [_event(4, "unchanged")],
    }
    predictions = {
        "p1": [_event(0, "added"), _event(1, "moved")],
        "p2": [_event(2, "unchanged", FAR)],
        "p3": [_event(4, "unchanged"), _event(5, "added")],
    }

    output = _run(tmp_path, ground_truth, predictions)
    aggregate = EventEvaluator().evaluate(predictions, ground_truth)

    rows = _event_rows(output)
    pair_rows = _pair_rows(output)
    for status in STATUSES:
        assert sum(1 for row in rows if row["gt_status"] == status) == (
            aggregate["per_status"][status]["fn"]
        ), status
        assert sum(1 for row in rows if row["pred_status"] == status) == (
            aggregate["per_status"][status]["fp"]
        ), status
    assert sum(int(row["tp"]) for row in pair_rows.values()) == sum(
        values["tp"] for values in aggregate["per_status"].values()
    )
    assert sum(int(row["fn"]) for row in pair_rows.values()) == sum(
        values["fn"] for values in aggregate["per_status"].values()
    )
    assert sum(int(row["fp"]) for row in pair_rows.values()) == sum(
        values["fp"] for values in aggregate["per_status"].values()
    )


def test_cli_writes_analysis_and_refuses_overwrite(tmp_path: Path) -> None:
    ground_truth = {"p1": [_event(0, "moved")], "p2": [_event(1, "unchanged")]}
    predictions = {"p1": [], "p2": [_event(1, "unchanged")]}
    manifest = _write_manifest(tmp_path, ground_truth)
    preds = _write_predictions(tmp_path, predictions)
    output = tmp_path / "cli_out"

    command = [sys.executable, str(CLI), "--predictions", str(preds),
               "--manifest", str(manifest), "--output-dir", str(output),
               "--iou-threshold", str(IOU)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False,
                               timeout=60, cwd=ROOT)
    assert completed.returncode == 0, completed.stderr
    assert "pairs=2 errors=1" in completed.stdout
    assert sorted(p.name for p in output.iterdir()) == [
        "event_errors.csv", "pair_results.csv", "selected_cases.json", "summary.json"
    ]

    again = subprocess.run(command, capture_output=True, text=True, check=False,
                           timeout=60, cwd=ROOT)
    assert again.returncode == 2
    assert "refusing" in again.stderr


def test_iou_threshold_is_required_by_the_api(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, {"p1": []})
    preds = _write_predictions(tmp_path, {"p1": []})

    with pytest.raises(TypeError, match="iou_threshold"):
        analyze_errors(preds, manifest, tmp_path / "out", manifest_loader=load_manifest)
    assert not (tmp_path / "out").exists()


def test_iou_threshold_of_zero_is_rejected_by_the_api(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, {"p1": []})
    preds = _write_predictions(tmp_path, {"p1": []})

    with pytest.raises(ValueError, match=r"iou_threshold must be finite and in \(0, 1\]"):
        analyze_errors(preds, manifest, tmp_path / "out", iou_threshold=0,
                       manifest_loader=load_manifest)
    assert not (tmp_path / "out").exists()


def test_iou_threshold_controls_misalignment(tmp_path: Path) -> None:
    ground_truth = {"p1": [_event(0, "unchanged", BOX)]}
    predictions = {"p1": [_event(0, "unchanged", SHIFTED)]}
    manifest = _write_manifest(tmp_path, ground_truth)

    loose = _run(tmp_path, ground_truth, predictions, manifest_path=manifest,
                 iou_threshold=0.5, out_name="loose")
    strict = _run(tmp_path, ground_truth, predictions, manifest_path=manifest,
                  iou_threshold=0.9, out_name="strict")

    assert _event_rows(loose) == []
    assert _pair_rows(loose)["p1"]["all_correct"] == "true"
    (error,) = _event_rows(strict)
    assert error["error_type"] == "misaligned_event"
    assert _summary(loose)["rules"]["event_iou_threshold"] == 0.5
    assert _summary(strict)["rules"]["event_iou_threshold"] == 0.9
    assert _summary(strict)["rules"]["event_iou_threshold_source"] == (
        "user-supplied iou_threshold"
    )


def test_cli_requires_iou_threshold(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, {"p1": []})
    preds = _write_predictions(tmp_path, {"p1": []})
    output = tmp_path / "cli_out"

    completed = subprocess.run(
        [sys.executable, str(CLI), "--predictions", str(preds),
         "--manifest", str(manifest), "--output-dir", str(output)],
        capture_output=True, text=True, check=False, timeout=60, cwd=ROOT,
    )

    assert completed.returncode == 2
    assert "--iou-threshold" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("value", ["0", "1.5", "-0.1", "nan", "abc"])
def test_cli_rejects_invalid_iou_threshold(tmp_path: Path, value: str) -> None:
    manifest = _write_manifest(tmp_path, {"p1": []})
    preds = _write_predictions(tmp_path, {"p1": []})
    output = tmp_path / "cli_out"

    completed = subprocess.run(
        [sys.executable, str(CLI), "--predictions", str(preds),
         "--manifest", str(manifest), "--output-dir", str(output),
         "--iou-threshold", value],
        capture_output=True, text=True, check=False, timeout=60, cwd=ROOT,
    )

    assert completed.returncode == 2
    assert "iou-threshold" in completed.stderr
    assert not output.exists()


def test_module_never_accesses_image_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ground_truth = {"p1": [_event(0, "moved")]}
    preds = _write_predictions(tmp_path, {"p1": []})
    manifest = {
        "schema_version": "1.0",
        "dataset_release": "fixture-v1",
        "split": "val",
        "pairs": [{"pair_id": "p1", "before_path": "photos/p1_before.png",
                   "after_path": "photos/p1_after.png"}],
        "ground_truth": ground_truth,
        "eligible_no_change_pair_ids": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    touched: list[str] = []

    def guard(name: str, real):
        def wrapper(self, *args, **kwargs):
            if self.suffix == ".png":
                touched.append(str(self))
                raise AssertionError(f"module accessed image path: {self}")
            return real(self, *args, **kwargs)

        return wrapper

    for name in ("stat", "resolve", "open", "is_file", "read_bytes"):
        monkeypatch.setattr(Path, name, guard(name, getattr(Path, name)))

    # The manifest dict is injected directly, so load_manifest (which stats) never runs.
    analyze_errors(preds, manifest_path, tmp_path / "study",
                   iou_threshold=IOU, manifest_loader=lambda _path: manifest)

    assert touched == []
    assert (tmp_path / "study" / "pair_results.csv").is_file()


def test_publish_failure_leaves_no_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ground_truth = {"p1": [_event(0, "moved")]}
    manifest = _write_manifest(tmp_path, ground_truth)
    preds = _write_predictions(tmp_path, {"p1": []})
    output = tmp_path / "study"

    class FailingPath(type(Path())):
        def rename(self, target):
            raise OSError("simulated publish failure")

    monkeypatch.setattr(error_analysis, "Path", FailingPath)

    with pytest.raises(OSError, match="simulated publish failure"):
        analyze_errors(preds, manifest, output, iou_threshold=IOU,
                       manifest_loader=load_manifest)

    assert not output.exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".study-")) == []
