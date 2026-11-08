import copy
import json
from math import inf, nan

import pytest

from deskcheck.evaluation.event_evaluator import EventEvaluator

BOX = [0, 0, 2, 2]


def event(status="unchanged", class_id=0, before=None, after=None):
    return {
        "class_id": class_id, "status": status,
        "before_box": (before if before is not None else BOX.copy())
        if status != "added" else None,
        "after_box": (after if after is not None else BOX.copy())
        if status != "removed" else None,
    }


def score(pred, truth):
    return EventEvaluator().evaluate({"p": pred}, {"p": truth})


@pytest.mark.parametrize("status", EventEvaluator.STATUSES)
def test_correct_event(status):
    result = score([event(status)], [event(status)])
    assert result["per_status"][status]["tp"] == 1
    assert result["per_status"][status]["f1"] == 1
    assert result["macro_f1"] == 0.25


def test_all_statuses_perfect():
    events = [event(s, i) for i, s in enumerate(EventEvaluator.STATUSES)]
    result = score(events, events)
    assert result["macro_f1"] == result["change_only_macro_f1"] == 1
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_wrong_status_counts_fp_and_fn():
    result = score([event("removed")], [event("moved")])["per_status"]
    assert result["removed"]["fp"] == 1
    assert result["moved"]["fn"] == 1
    assert sum(r["tp"] for r in result.values()) == 0


def test_missing_prediction():
    result = score([], [event("added")])["per_status"]["added"]
    assert result["fn"] == result["support"] == 1
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_extra_prediction():
    result = score([event("added")], [])["per_status"]["added"]
    assert result["fp"] == 1 and result["support"] == 0


def test_wrong_class():
    result = score([event(class_id=1)], [event(class_id=0)])["per_status"]["unchanged"]
    assert result["fp"] == result["fn"] == 1


def test_wrong_geometry_even_when_status_correct():
    prediction = event(after=[10, 10, 12, 12])
    result = score([prediction], [event()])["per_status"]["unchanged"]
    assert result["tp"] == 0 and result["fp"] == result["fn"] == 1


@pytest.mark.parametrize("shift, correct", [(1, True), (1.01, False)])
def test_iou_boundary(shift, correct):
    # Equal-area height-one boxes, width three: IoU is 0.5 at shift=1.
    gt = event("added", after=[0, 0, 3, 1])
    pred = event("added", after=[shift, 0, shift + 3, 1])
    result = score([pred], [gt])["per_status"]["added"]
    assert result["tp"] == int(correct)


def test_all_empty_scenes_have_zero_macro_not_perfect():
    result = score([], [])
    assert result["macro_f1"] == 0
    assert all(r["support"] == 0 for r in result["per_status"].values())


def test_aggregate_counts_not_pair_f1_average():
    evaluator = EventEvaluator()
    result = evaluator.evaluate(
        {"a": [event("added")], "b": []},
        {"a": [event("added")], "b": [event("added")]},
    )["per_status"]["added"]
    assert result["tp"] == result["fn"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("value", [0, -0.1, 1.1, nan, inf])
def test_invalid_iou(value):
    with pytest.raises(ValueError):
        EventEvaluator(value)


@pytest.mark.parametrize("value", [None, "0.5", True])
def test_invalid_iou_type(value):
    with pytest.raises(TypeError):
        EventEvaluator(value)


def test_missing_pair_is_not_empty_prediction():
    with pytest.raises(ValueError, match="match exactly"):
        EventEvaluator().evaluate({}, {"p": []})


def test_no_pairs_rejected():
    with pytest.raises(ValueError):
        EventEvaluator().evaluate({}, {})


@pytest.mark.parametrize("side", ["prediction", "truth"])
def test_duplicate_class_rejected(side):
    duplicate = [event(), event()]
    with pytest.raises(ValueError, match="Duplicate"):
        score(duplicate if side == "prediction" else [], duplicate if side == "truth" else [])


@pytest.mark.parametrize("field, value", [
    ("status", "unknown"), ("class_id", -1), ("class_id", True),
    ("before_box", None), ("after_box", [0, 0, 0, 1]),
    ("after_box", [-1, 0, 1, 1]), ("after_box", [0, 0, nan, 1]),
    ("after_box", [0, 0, inf, 1]), ("after_box", [0, 1]),
])
def test_invalid_events(field, value):
    invalid = event()
    invalid[field] = value
    with pytest.raises(ValueError):
        score([invalid], [])


def test_required_fields():
    with pytest.raises(ValueError, match="required fields"):
        score([{"status": "added"}], [])


@pytest.mark.parametrize("bad", [None, 1, "bad"])
def test_event_type(bad):
    with pytest.raises(TypeError):
        score([bad], [])


def test_event_list_type():
    with pytest.raises(TypeError):
        score({}, [])


def test_mapping_type():
    with pytest.raises(TypeError):
        EventEvaluator().evaluate([], {})


def test_no_mutation_and_order_independence():
    events = [event("moved", 2), event("added", 0)]
    original = copy.deepcopy(events)
    assert score(events, events) == score(events[::-1], events[::-1])
    assert events == original


def test_shared_single_side_mismatch_and_no_shared_side():
    result = score([event("added")], [event("removed")])["per_status"]
    assert result["added"]["fp"] == result["removed"]["fn"] == 1


def test_repeated_calls_are_stateless():
    evaluator = EventEvaluator()
    evaluator.evaluate({"p": [event()]}, {"p": [event()]})
    assert evaluator.evaluate({"p": []}, {"p": []})["macro_f1"] == 0
