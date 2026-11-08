import copy
import json

import pytest

from deskcheck.evaluation.no_change_evaluator import NoChangeEvaluator


def test_no_alarms():
    result = NoChangeEvaluator().evaluate({"p": [{"status": "unchanged"}]}, ["p"])
    assert result["false_alarm_rate"] == 0
    assert result["eligible_pair_count"] == 1


@pytest.mark.parametrize("status", ["added", "removed", "moved"])
def test_each_change_is_alarm(status):
    result = NoChangeEvaluator().evaluate({"p": [{"status": status}]}, ["p"])
    assert result["false_alarm_rate"] == 1


def test_multiple_changes_in_one_pair_count_once():
    result = NoChangeEvaluator().evaluate(
        {"p": [{"status": "added"}, {"status": "moved"}]}, ["p"]
    )
    assert result["false_alarm_pair_count"] == 1


def test_fraction_and_sorted_ids():
    results = {"c": [], "b": [{"status": "removed"}], "a": [{"status": "moved"}]}
    result = NoChangeEvaluator().evaluate(results, ["c", "b", "a"])
    assert result["false_alarm_rate"] == pytest.approx(2 / 3)
    assert result["false_alarm_pair_ids"] == ["a", "b"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_ignores_noneligible_pairs():
    result = NoChangeEvaluator().evaluate({"p": [], "other": [{"status": "moved"}]}, ["p"])
    assert result["eligible_pair_count"] == 1 and result["false_alarm_rate"] == 0


def test_explicit_empty_result_is_no_alarm():
    assert NoChangeEvaluator().evaluate({"p": []}, ["p"])["false_alarm_rate"] == 0


def test_missing_result_is_error():
    with pytest.raises(ValueError, match="Missing eligible"):
        NoChangeEvaluator().evaluate({}, ["p"])


def test_empty_eligibility_is_error():
    with pytest.raises(ValueError):
        NoChangeEvaluator().evaluate({}, [])


def test_duplicate_eligibility_is_error():
    with pytest.raises(ValueError, match="Duplicate"):
        NoChangeEvaluator().evaluate({"p": []}, ["p", "p"])


@pytest.mark.parametrize("ids", [[""], [None], [1]])
def test_invalid_eligible_id(ids):
    with pytest.raises(ValueError):
        NoChangeEvaluator().evaluate({}, ids)


@pytest.mark.parametrize("ids", [None, "p", {"p"}])
def test_eligible_container_type(ids):
    with pytest.raises(TypeError):
        NoChangeEvaluator().evaluate({}, ids)


def test_results_container_type():
    with pytest.raises(TypeError):
        NoChangeEvaluator().evaluate([], ["p"])


@pytest.mark.parametrize("events", [None, {}])
def test_events_container_type(events):
    with pytest.raises(TypeError):
        NoChangeEvaluator().evaluate({"p": events}, ["p"])


@pytest.mark.parametrize("event", [None, "bad"])
def test_event_type(event):
    with pytest.raises(TypeError):
        NoChangeEvaluator().evaluate({"p": [event]}, ["p"])


@pytest.mark.parametrize("event", [{}, {"status": "bad"}, {"status": None}])
def test_invalid_status(event):
    with pytest.raises(ValueError):
        NoChangeEvaluator().evaluate({"p": [event]}, ["p"])


def test_invalid_event_after_alarm_is_still_validated():
    with pytest.raises(ValueError):
        NoChangeEvaluator().evaluate({"p": [{"status": "moved"}, {}]}, ["p"])


def test_no_mutation_and_statelessness():
    evaluator = NoChangeEvaluator()
    results, ids = {"p": [{"status": "moved"}]}, ["p"]
    snapshot = copy.deepcopy(results)
    evaluator.evaluate(results, ids)
    assert results == snapshot and ids == ["p"]
    assert evaluator.evaluate({"p": []}, ["p"])["false_alarm_rate"] == 0
