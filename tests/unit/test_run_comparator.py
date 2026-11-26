import hashlib
import json

import pytest

from deskcheck.evaluation.evaluation_runner import EvaluationRunner
from deskcheck.evaluation.failure_record import (
    build_failure_record,
    build_rerun_link,
    write_failure_record,
)
from deskcheck.evaluation.run_comparator import compare_runs, write_comparison

BOX = [0, 0, 2, 2]


def event(status="unchanged", class_id=0):
    return {
        "class_id": class_id,
        "status": status,
        "before_box": None if status == "added" else BOX,
        "after_box": None if status == "removed" else BOX,
    }


def pair(pair_id, image_dir=None):
    if image_dir is None:
        return {
            "pair_id": pair_id,
            "before_path": f"{pair_id}-before.png",
            "after_path": f"{pair_id}-after.png",
        }
    image_dir.mkdir(parents=True, exist_ok=True)
    before_bytes = f"{pair_id}-before-image".encode()
    after_bytes = f"{pair_id}-after-image".encode()
    before_path = image_dir / f"{pair_id}-before.png"
    after_path = image_dir / f"{pair_id}-after.png"
    before_path.write_bytes(before_bytes)
    after_path.write_bytes(after_bytes)
    return {
        "pair_id": pair_id,
        "before_path": before_path,
        "after_path": after_path,
        "before_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "after_sha256": hashlib.sha256(after_bytes).hexdigest(),
    }


class FixtureService:
    def __init__(self, predictions):
        self.predictions = predictions

    def compare(self, before_path, after_path, pair_id):
        return {"pair_id": pair_id, "records": self.predictions[pair_id]}


def provenance(model_index, *, split="val", **overrides):
    value = {
        "mode": "synthetic-test-fixture",
        "run_id": f"run-{model_index}",
        "checkpoint_sha256": str(model_index) * 64,
        "config_sha256": str(model_index + 2) * 64,
        "model_origin": "synthetic-test-fixture",
        "confidence_threshold": 0.2,
        "movement_threshold": 0.1,
        "nms_iou_threshold": 0.35,
        "imgsz": 640,
        "rules_version": "synthetic-v1",
        "split": split,
        "dataset_release": "synthetic-dataset",
        "manifest_sha256": "a" * 64,
        "source_annotations_sha256": "b" * 64,
        "source_split_manifest_sha256": "c" * 64,
        "python_version": "test",
        "package_versions": {"numpy": "test"},
        "command": ["synthetic-test"],
        "source_revision": "synthetic-source",
        "source_dirty": False,
        "source_dirty_paths": [],
    }
    value.update(overrides)
    return value


def create_runs(tmp_path, *, control_overrides=None, treatment_overrides=None):
    image_dir = tmp_path / "synthetic-images"
    pairs = [pair("changed", image_dir), pair("no-change", image_dir)]
    truth = {
        "changed": [event("added")],
        "no-change": [event("unchanged")],
    }
    control_predictions = {
        "changed": [event("added")],
        "no-change": [event("unchanged")],
    }
    treatment_predictions = {
        "changed": [],
        "no-change": [event("moved")],
    }
    paths = []
    for index, (predictions, overrides) in enumerate(
        (
            (control_predictions, control_overrides or {}),
            (treatment_predictions, treatment_overrides or {}),
        ),
        start=1,
    ):
        output = tmp_path / f"arm-{index}"
        EvaluationRunner(FixtureService(predictions)).run(
            pairs,
            truth,
            ["no-change"],
            output,
            metadata=provenance(index, **overrides),
        )
        paths.append(output)
    return paths


def test_compare_saved_runs_reports_aggregate_pair_and_no_change_results(tmp_path):
    control, treatment = create_runs(tmp_path)
    report = compare_runs(control, treatment)

    assert report["split"] == "val"
    assert report["pair_count"] == 2
    assert report["aggregate"]["per_status"]["added"]["fn"]["treatment_minus_control"] == 1
    event_false_positives = report["aggregate"]["no_change"]["false_positive_event_count"]
    assert {key: value for key, value in event_false_positives.items() if key != "role"} == {
        "control": 0,
        "treatment": 1,
        "treatment_minus_control": 1,
    }
    assert event_false_positives["role"] == "secondary"
    assert report["aggregate"]["no_change"]["false_alarm_pair_rate"]["treatment"] == 1
    assert report["per_pair"]["changed"]["per_status"]["added"]["fn"]["treatment"] == 1
    assert report["control"]["checkpoint_sha256"] != report["treatment"]["checkpoint_sha256"]


def test_primary_metric_is_explicit_and_secondary_metrics_are_labelled(tmp_path):
    control, treatment = create_runs(tmp_path)
    report = compare_runs(control, treatment)

    primary = report["primary_metric"]
    assert primary["name"] == "change_only_macro_f1"
    assert primary["role"] == "primary"
    assert primary["decision_date"] == "2026-11-26"
    assert primary["control"] == pytest.approx(1 / 3)
    assert primary["treatment"] == pytest.approx(0.0)
    assert primary["treatment_minus_control"] == pytest.approx(-1 / 3)
    assert report["metric_roles"]["primary"] == ["change_only_macro_f1"]
    assert report["aggregate"]["event_metrics"]["change_only_macro_f1"]["role"] == "primary"
    assert report["aggregate"]["event_metrics"]["macro_f1"]["role"] == "secondary"
    assert "Absent change statuses" in report["primary_metric_note"]

    control_metadata = json.loads((control / "run_metadata.json").read_text())
    assert control_metadata["primary_downstream_metric"]["name"] == "change_only_macro_f1"
    assert control_metadata["primary_downstream_metric"]["role"] == "primary"


def test_comparison_writes_json_and_human_summary_without_overwrite(tmp_path):
    control, treatment = create_runs(tmp_path)
    output = tmp_path / "comparison"
    report = compare_runs(control, treatment)

    written = write_comparison(report, output)
    saved = json.loads(written.read_text(encoding="utf-8"))
    assert saved == report
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "false_positive_event_count" in summary
    assert "pairs with at least one predicted change divided by eligible no-change pairs" in summary
    assert "**change_only_macro_f1** | **primary**" in summary
    assert "| secondary |" in summary
    assert "## Linked failures" in summary
    with pytest.raises(FileExistsError):
        write_comparison(report, output)


def test_comparator_surfaces_linked_failures_without_accepting_them(tmp_path):
    failure = build_failure_record(
        run_id="failed-run-synthetic",
        reason="synthetic evaluation failure",
        exception_type="RuntimeError",
        context={"stage": "per_pair", "pair_id": "changed", "image_id": None},
        input_identities={
            "manifest_sha256": "a" * 64,
            "checkpoint_sha256": "2" * 64,
            "config_sha256": "4" * 64,
            "dataset_release": "synthetic-dataset",
            "split": "val",
        },
        source_identity={
            "source_revision": "synthetic-source",
            "source_dirty": False,
            "source_dirty_paths": [],
        },
        evaluation_config={"confidence_threshold": 0.2},
        command=["synthetic"],
    )
    failure_path = write_failure_record(failure, tmp_path / "failure")
    link = build_rerun_link(failure_path, "synthetic retry after failure")

    image_dir = tmp_path / "synthetic-images"
    pairs = [pair("changed", image_dir), pair("no-change", image_dir)]
    truth = {"changed": [event("added")], "no-change": [event("unchanged")]}
    predictions = {"changed": [event("added")], "no-change": [event("unchanged")]}
    control = tmp_path / "control"
    EvaluationRunner(FixtureService(predictions)).run(
        pairs, truth, ["no-change"], control, metadata=provenance(1)
    )
    treatment = tmp_path / "treatment"
    EvaluationRunner(FixtureService({"changed": [], "no-change": [event("unchanged")]})).run(
        pairs,
        truth,
        ["no-change"],
        treatment,
        metadata=provenance(2),
        rerun_of=[link],
    )

    report = compare_runs(control, treatment)
    assert report["linked_failures"]["control"] == []
    assert report["linked_failures"]["treatment"] == [link]
    assert "owner decision" in report["linked_failures"]["note"]
    summary = (write_comparison(report, tmp_path / "comparison")).parent / "summary.md"
    text = summary.read_text(encoding="utf-8")
    assert "failed-run-synthetic" in text
    assert "synthetic retry after failure" in text


def test_comparator_rejects_a_changed_primary_metric_claim(tmp_path):
    control, treatment = create_runs(tmp_path)
    metrics_path = treatment / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["events"]["primary_metric"]["name"] = "macro_f1"
    metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(ValueError, match="primary metric"):
        compare_runs(control, treatment)


def test_comparator_rejects_a_run_without_a_recorded_primary_metric(tmp_path):
    control, treatment = create_runs(tmp_path)
    metadata_path = treatment / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["primary_downstream_metric"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="primary metric"):
        compare_runs(control, treatment)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"split": "test"}, "frozen split"),
        ({"manifest_sha256": "b" * 64}, "frozen manifest_sha256"),
        ({"confidence_threshold": 0.3}, "frozen confidence_threshold"),
        ({"source_dirty": True, "source_dirty_paths": ["changed.py"]}, "frozen source_dirty"),
    ],
)
def test_comparison_rejects_mixed_inputs_or_evaluation_rules(tmp_path, overrides, message):
    control, treatment = create_runs(tmp_path, treatment_overrides=overrides)
    with pytest.raises(ValueError, match=message):
        compare_runs(control, treatment)


def test_comparison_rejects_same_checkpoint_identity(tmp_path):
    control, treatment = create_runs(tmp_path)
    treatment_metadata_path = treatment / "run_metadata.json"
    metadata = json.loads(treatment_metadata_path.read_text())
    metadata["provenance"]["checkpoint_sha256"] = "1" * 64
    treatment_metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="distinct checkpoint bytes"):
        compare_runs(control, treatment)


def test_comparison_rejects_inconsistent_saved_metrics(tmp_path):
    control, treatment = create_runs(tmp_path)
    metrics_path = treatment / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["events"]["per_status"]["added"]["tp"] = 99
    metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(ValueError, match="do not match"):
        compare_runs(control, treatment)


def test_comparison_rejects_pair_inventory_mismatch(tmp_path):
    control, treatment = create_runs(tmp_path)
    metadata_path = treatment / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["pairs"][0]["before_path"] = "different-before.png"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="same image paths"):
        compare_runs(control, treatment)
