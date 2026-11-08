"""Runner tests inject a fake service but use the real PR4/PR5 evaluators."""

import copy
import json
from pathlib import Path

import pytest

from deskcheck.evaluation.evaluation_runner import EvaluationRunner


def event(status="unchanged"):
    return {
        "class_id": 0, "status": status,
        "before_box": None if status == "added" else [0, 0, 2, 2],
        "after_box": None if status == "removed" else [0, 0, 2, 2],
    }


def pair(pair_id):
    return {"pair_id": pair_id, "before_path": f"{pair_id}_b.jpg",
            "after_path": f"{pair_id}_a.jpg"}


class FakeComparisonService:
    def __init__(self, results, fail_on=None):
        self.results = results
        self.fail_on = fail_on
        self.calls = []

    def compare(self, before_path, after_path, pair_id):
        self.calls.append((before_path, after_path, pair_id))
        if pair_id == self.fail_on:
            raise OSError("simulated inference failure")
        return {"pair_id": pair_id, "records": self.results[pair_id]}


def run(tmp_path, *, predictions=None, truth=None, pairs=None, eligible=None, metadata=None):
    predictions = {"p": [event()]} if predictions is None else predictions
    truth = {"p": [event()]} if truth is None else truth
    pairs = [pair("p")] if pairs is None else pairs
    eligible = ["p"] if eligible is None else eligible
    service = FakeComparisonService(predictions)
    result = EvaluationRunner(service).run(pairs, truth, eligible, tmp_path / "result", metadata)
    return result, service


def test_success_outputs_actual_evaluator_metrics_and_metadata(tmp_path):
    result, service = run(tmp_path, metadata={"split": "synthetic", "checkpoint_sha256": None})
    assert len(service.calls) == 1
    assert all(isinstance(path, Path) for path in service.calls[0][:2])
    assert result["metrics"]["events"]["per_status"]["unchanged"]["tp"] == 1
    assert result["metrics"]["events"]["macro_f1"] == 0.25
    assert result["metrics"]["no_change"]["false_alarm_rate"] == 0
    assert result["run_metadata"]["provenance"]["split"] == "synthetic"
    for filename, key in [("predictions.json", "predictions"), ("metrics.json", "metrics"),
                          ("run_metadata.json", "run_metadata")]:
        assert json.loads((tmp_path / "result" / filename).read_text()) == result[key]


def test_complete_inventory_sorted_and_calls_once(tmp_path):
    predictions = {"b": [event("added")], "a": [event()]}
    truth = copy.deepcopy(predictions)
    result, service = run(tmp_path, predictions=predictions, truth=truth,
                          pairs=[pair("b"), pair("a")], eligible=["a"])
    assert [call[2] for call in service.calls] == ["a", "b"]
    assert result["run_metadata"]["pair_count"] == 2
    assert set(result["predictions"]) == {"a", "b"}


def test_false_alarm_reaches_no_change_evaluator(tmp_path):
    result, _ = run(tmp_path, predictions={"p": [event("moved")]})
    assert result["metrics"]["no_change"]["false_alarm_rate"] == 1
    assert result["metrics"]["events"]["per_status"]["moved"]["fp"] == 1


def test_empty_prediction_is_valid_not_missing(tmp_path):
    result, _ = run(tmp_path, predictions={"p": []})
    assert result["metrics"]["events"]["per_status"]["unchanged"]["fn"] == 1
    assert result["metrics"]["no_change"]["false_alarm_rate"] == 0


def test_inference_failure_stops_without_output_and_preserves_cause(tmp_path):
    service = FakeComparisonService({"a": [event()], "b": [event()]}, fail_on="b")
    with pytest.raises(RuntimeError, match="'b'") as error:
        EvaluationRunner(service).run([pair("b"), pair("a")],
                                     {"a": [event()], "b": [event()]}, ["a"], tmp_path / "result")
    assert isinstance(error.value.__cause__, OSError)
    assert [call[2] for call in service.calls] == ["a", "b"]
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("invalid", [None, [], {"records": []},
                                      {"pair_id": "wrong", "records": []},
                                      {"pair_id": "p"},
                                      {"pair_id": "p", "records": {}},
                                      {"pair_id": "p", "records": [{"status": "bad"}]}])
def test_bad_service_output_is_pair_context_error(tmp_path, invalid):
    class BadService:
        def compare(self, **kwargs):
            return invalid
    with pytest.raises(RuntimeError, match="'p'"):
        EvaluationRunner(BadService()).run([pair("p")], {"p": [event()]}, ["p"],
                                          tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_nonfinite_extra_record_field_is_rejected(tmp_path):
    bad = event()
    bad["distance_norm"] = float("nan")
    with pytest.raises(RuntimeError):
        run(tmp_path, predictions={"p": [bad]})
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("pairs", [[], [pair("p"), pair("p")], [{}],
                                   [{**pair("p"), "pair_id": ""}],
                                   [{**pair("p"), "before_path": ""}]])
def test_invalid_pair_preflight_without_inference(tmp_path, pairs):
    service = FakeComparisonService({})
    with pytest.raises(ValueError):
        EvaluationRunner(service).run(pairs, {"p": [event()]}, ["p"], tmp_path / "result")
    assert service.calls == []


@pytest.mark.parametrize("pairs", [None, {}, [None], [{**pair("p"), "before_path": None}]])
def test_invalid_pair_types(tmp_path, pairs):
    with pytest.raises(TypeError):
        run(tmp_path, pairs=pairs if pairs is not None else {})


@pytest.mark.parametrize("truth, eligible", [
    ({"other": [event()]}, ["other"]),
    ({"p": [{"status": "bad"}]}, ["p"]),
    ({"p": [event()]}, []),
    ({"p": [event()]}, ["p", "p"]),
    ({"p": [event()]}, ["missing"]),
    ({"p": [event("moved")]}, ["p"]),
    ({"p": []}, ["p"]),
])
def test_bad_truth_or_eligibility_fails_before_service(tmp_path, truth, eligible):
    service = FakeComparisonService({})
    with pytest.raises(ValueError):
        EvaluationRunner(service).run([pair("p")], truth, eligible, tmp_path / "result")
    assert service.calls == []


def test_existing_output_is_never_overwritten(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep")
    service = FakeComparisonService({"p": [event()]})
    with pytest.raises(FileExistsError):
        EvaluationRunner(service).run([pair("p")], {"p": [event()]}, ["p"], target)
    assert marker.read_text() == "keep" and service.calls == []


def test_publish_failure_cleans_staging(tmp_path, monkeypatch):
    original = Path.write_text
    def fail_metrics(path, *args, **kwargs):
        if path.name == "metrics.json":
            raise OSError("disk full")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", fail_metrics)
    with pytest.raises(OSError, match="disk full"):
        run(tmp_path)
    assert not (tmp_path / "result").exists()
    assert list(tmp_path.glob(".result-*")) == []


def test_inputs_are_not_mutated(tmp_path):
    pairs, truth, ids = [pair("p")], {"p": [event()]}, ["p"]
    metadata = {"settings": {"confidence": 0.5}}
    before = copy.deepcopy((pairs, truth, ids, metadata))
    run(tmp_path, pairs=pairs, truth=truth, eligible=ids, metadata=metadata)
    assert (pairs, truth, ids, metadata) == before


def test_reused_service_buffers_are_snapshotted(tmp_path):
    class ReusingService:
        def __init__(self):
            self.shared = [event()]
        def compare(self, before_path, after_path, pair_id):
            self.shared[0]["status"] = "unchanged" if pair_id == "a" else "moved"
            return {"pair_id": pair_id, "records": self.shared}
    result = EvaluationRunner(ReusingService()).run(
        [pair("a"), pair("b")], {"a": [event()], "b": [event("moved")]},
        ["a"], tmp_path / "result",
    )
    assert result["predictions"]["a"][0]["status"] == "unchanged"
    assert result["predictions"]["b"][0]["status"] == "moved"


@pytest.mark.parametrize("service", [None, object()])
def test_service_requires_compare(service):
    with pytest.raises(TypeError):
        EvaluationRunner(service)


@pytest.mark.parametrize("metadata", [[1], {"value": float("inf")}, {"path": Path("x")}])
def test_bad_metadata_rejected_before_calls(tmp_path, metadata):
    service = FakeComparisonService({})
    with pytest.raises((TypeError, ValueError)):
        EvaluationRunner(service).run([pair("p")], {"p": [event()]}, ["p"],
                                     tmp_path / "result", metadata)
    assert service.calls == []
