"""P1: audited annotations through conversion to the real evaluation runner."""

import copy
import json
from pathlib import Path

import pytest

from deskcheck.data.dataset_catalog import load_catalog
from deskcheck.data.dataset_validator import audit
from deskcheck.evaluation.annotation_converter import convert_dataset
from deskcheck.evaluation.evaluation_runner import EvaluationRunner

TEST_TAU = 0.02  # Synthetic verification only; not approval of the project tau.


def _box(class_id, moved=False):
    x = 10 + class_id * 80 + (24 if moved else 0)
    return [x, 10, x + 40, 50]


def _annotation(pair_id, group_id, kind):
    before_classes = range(6) if kind == "control" else range(5)
    after_classes = range(6) if kind == "control" else (0, 2, 3, 4, 5)
    statuses = {c: "unchanged" for c in range(6)}
    if kind == "change":
        statuses.update({1: "removed", 2: "moved", 5: "added"})
    if kind == "empty":
        before_classes, after_classes, statuses = (), (), {}
    before = [
        {"object_id": f"b{c}", "class_id": c, "bbox_xyxy": _box(c)}
        for c in before_classes
    ]
    after = [
        {"object_id": f"a{c}", "class_id": c,
         "bbox_xyxy": _box(c, moved=(kind == "change" and c == 2))}
        for c in after_classes
    ]
    before_ids = {o["class_id"]: o["object_id"] for o in before}
    after_ids = {o["class_id"]: o["object_id"] for o in after}
    return {
        "schema_version": "1.0", "pair_id": pair_id,
        "capture_group_id": group_id,
        "before_image_id": f"{pair_id}_b", "after_image_id": f"{pair_id}_a",
        "before_image_path": f"raw/{group_id}/{pair_id}_b.png",
        "after_image_path": f"raw/{group_id}/{pair_id}_a.png",
        "width": 640, "height": 480, "rules_version": "p1-test-only",
        "independently_captured_no_change": kind == "control",
        "manual_before_objects": before, "manual_after_objects": after,
        "correspondences": [
            {"class_id": c, "before_object_id": before_ids[c],
             "after_object_id": after_ids[c]}
            for c in sorted(before_ids.keys() & after_ids.keys())
        ],
        "events": [
            {"class_id": c, "status": status,
             "before_object_id": before_ids.get(c),
             "after_object_id": after_ids.get(c)}
            for c, status in statuses.items()
        ],
    }


@pytest.fixture
def catalog(tmp_path):
    records = []
    splits = {}
    for split_name in ("train", "val", "test"):
        group = f"g_{split_name}"
        splits[split_name] = []
        for kind in ("change", "control"):
            pair_id = f"{split_name}_{kind}"
            records.append(_annotation(pair_id, group, kind))
            splits[split_name].append(pair_id)
    for kind, pair_id in (("control", "val_control_2"), ("empty", "val_empty")):
        records.append(_annotation(pair_id, "g_val", kind))
        splits["val"].append(pair_id)
    annotations = tmp_path / "pairs.jsonl"
    manifest = tmp_path / "splits.json"
    annotations.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )
    manifest.write_text(json.dumps({
        "schema_version": "1.0", "seed": 42, "dataset_release": "p1-synthetic",
        "splits": splits,
    }), encoding="utf-8")
    return load_catalog(annotations, manifest)


def _inputs(catalog):
    pairs, manifest = catalog
    report = audit(pairs, manifest, movement_threshold=TEST_TAU)
    assert report.ok, report.errors
    assert not report.warnings, report.warnings
    ids = list(manifest.splits["val"])
    selected = {pid: pairs[pid] for pid in ids}
    inventory = [
        {"pair_id": pid, "before_path": selected[pid].before_image_path,
         "after_path": selected[pid].after_image_path}
        for pid in ids
    ]
    eligible = [pid for pid in ids if selected[pid].independently_captured_no_change]
    return inventory, convert_dataset(selected), eligible


def _predictions():
    # Independent literal oracle: deliberately does not call annotation helpers/converter.
    change = [
        {"class_id": 0, "status": "unchanged",
         "before_box": [10, 10, 50, 50], "after_box": [10, 10, 50, 50]},
        {"class_id": 1, "status": "removed",
         "before_box": [90, 10, 130, 50], "after_box": None},
        {"class_id": 2, "status": "moved",
         "before_box": [170, 10, 210, 50], "after_box": [194, 10, 234, 50]},
        {"class_id": 3, "status": "unchanged",
         "before_box": [250, 10, 290, 50], "after_box": [250, 10, 290, 50]},
        {"class_id": 4, "status": "unchanged",
         "before_box": [330, 10, 370, 50], "after_box": [330, 10, 370, 50]},
        {"class_id": 5, "status": "added",
         "before_box": None, "after_box": [410, 10, 450, 50]},
    ]
    boxes = [
        [10, 10, 50, 50], [90, 10, 130, 50], [170, 10, 210, 50],
        [250, 10, 290, 50], [330, 10, 370, 50], [410, 10, 450, 50],
    ]
    control = [
        {"class_id": c, "status": "unchanged",
         "before_box": list(box), "after_box": list(box)}
        for c, box in enumerate(boxes)
    ]
    return {"val_change": change, "val_control": control,
            "val_control_2": copy.deepcopy(control), "val_empty": []}


class FakeComparisonService:
    def __init__(self, predictions, fail_on=None):
        self.predictions = predictions
        self.fail_on = fail_on
        self.calls = []

    def compare(self, before_path, after_path, pair_id):
        self.calls.append((before_path, after_path, pair_id))
        if pair_id == self.fail_on:
            raise OSError("P1 simulated service failure")
        return {"pair_id": pair_id, "records": copy.deepcopy(self.predictions[pair_id])}


def test_p1_audited_jsonl_to_runner_outputs(catalog, tmp_path):
    inventory, truth, eligible = _inputs(catalog)
    expected = _predictions()
    assert set(truth) == set(expected)
    for pid, events in truth.items():
        assert [{k: event[k] for k in
                 ("class_id", "status", "before_box", "after_box")}
                for event in events] == expected[pid]
        assert all(event["pair_id"] == pid for event in events)
    service = FakeComparisonService(expected)
    target = tmp_path / "result"
    result = EvaluationRunner(service).run(
        inventory, truth, eligible, target,
        metadata={"split": "val", "dataset_release": "p1-synthetic",
                  "rules_version": "p1-test-only", "movement_threshold": TEST_TAU,
                  "service": "fake; no image decoding or model inference"},
    )
    assert [call[2] for call in service.calls] == sorted(expected)
    assert all(isinstance(path, Path) for call in service.calls for path in call[:2])
    assert result["predictions"] == expected
    assert result["predictions"]["val_empty"] == []
    metrics = result["metrics"]
    assert metrics["events"]["pair_count"] == 4
    assert metrics["events"]["macro_f1"] == 1.0
    assert metrics["events"]["change_only_macro_f1"] == 1.0
    for status, tp in {"added": 1, "removed": 1, "moved": 1, "unchanged": 15}.items():
        counts = metrics["events"]["per_status"][status]
        assert (counts["tp"], counts["fp"], counts["fn"]) == (tp, 0, 0)
    assert metrics["no_change"]["eligible_pair_count"] == 2
    assert metrics["no_change"]["false_alarm_pair_count"] == 0
    assert metrics["no_change"]["false_alarm_rate"] == 0
    metadata = result["run_metadata"]
    assert metadata["status"] == "complete"
    assert metadata["pair_count"] == 4
    assert metadata["eligible_no_change_pair_ids"] == sorted(eligible)
    assert metadata["provenance"]["rules_version"] == "p1-test-only"
    files = {"predictions.json": "predictions", "metrics.json": "metrics",
             "run_metadata.json": "run_metadata"}
    assert {p.name for p in target.iterdir()} == set(files)
    for filename, key in files.items():
        assert json.loads((target / filename).read_text(encoding="utf-8")) == result[key]


def test_p1_one_wrong_status_changes_event_and_alarm_counts(catalog, tmp_path):
    inventory, truth, eligible = _inputs(catalog)
    predictions = _predictions()
    predictions["val_control"][0]["status"] = "moved"
    result = EvaluationRunner(FakeComparisonService(predictions)).run(
        inventory, truth, eligible, tmp_path / "wrong-status",
    )
    events = result["metrics"]["events"]["per_status"]
    assert (events["moved"]["tp"], events["moved"]["fp"], events["moved"]["fn"]) == (1, 1, 0)
    assert (events["unchanged"]["tp"], events["unchanged"]["fp"],
            events["unchanged"]["fn"]) == (14, 0, 1)
    alarms = result["metrics"]["no_change"]
    assert alarms["eligible_pair_count"] == 2
    assert alarms["false_alarm_pair_count"] == 1
    assert alarms["false_alarm_rate"] == 0.5
    assert alarms["false_alarm_pair_ids"] == ["val_control"]
    assert result["predictions"]["val_control_2"] == _predictions()["val_control_2"]


def test_p1_service_failure_does_not_publish(catalog, tmp_path):
    inventory, truth, eligible = _inputs(catalog)
    service = FakeComparisonService(_predictions(), fail_on="val_control")
    target = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="val_control") as error:
        EvaluationRunner(service).run(inventory, truth, eligible, target)
    assert isinstance(error.value.__cause__, OSError)
    assert [call[2] for call in service.calls] == ["val_change", "val_control"]
    assert not target.exists()
    assert list(tmp_path.glob(".failed-*")) == []


def test_p1_invalid_no_change_annotation_is_rejected(catalog, tmp_path):
    pairs, manifest = copy.deepcopy(catalog)
    pairs["val_control"].events[0]["status"] = "moved"
    report = audit(pairs, manifest, movement_threshold=TEST_TAU)
    assert not report.ok
    assert any("independent no-change pair contains change events" in e for e in report.errors)
    with pytest.raises(ValueError, match="no-change pair contains change"):
        convert_dataset({"val_control": pairs["val_control"]})
    assert not (tmp_path / "invalid").exists()


def test_p1_wrong_eligibility_fails_before_inference(catalog, tmp_path):
    inventory, truth, eligible = _inputs(catalog)
    service = FakeComparisonService(_predictions())
    with pytest.raises(ValueError, match="Eligible no-change ground truth contains change"):
        EvaluationRunner(service).run(
            inventory, truth, [*eligible, "val_change"], tmp_path / "invalid-eligibility",
        )
    assert service.calls == []
    assert not (tmp_path / "invalid-eligibility").exists()


def test_p1_empty_scene_is_not_an_eligible_annotated_control(catalog, tmp_path):
    inventory, truth, eligible = _inputs(catalog)
    assert truth["val_empty"] == []
    service = FakeComparisonService(_predictions())
    with pytest.raises(ValueError, match="require annotated unchanged objects"):
        EvaluationRunner(service).run(
            inventory, truth, [*eligible, "val_empty"], tmp_path / "empty-control",
        )
    assert service.calls == []
    assert not (tmp_path / "empty-control").exists()
