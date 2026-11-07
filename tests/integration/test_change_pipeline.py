"""Run actual B modules together, without a detector or UI dependency."""

import json

import pytest

from deskcheck.changes.change_analyzer import ChangeAnalyzer
from deskcheck.detection.detection_filter import DetectionFilter
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection
from deskcheck.matching.object_matcher import ObjectMatcher


def detection(name, class_id, confidence=0.9, box=None):
    return Detection(name, class_id, confidence, box or BoundingBox(0, 0, 2, 2))


def run_pipeline(before, after, *, confidence=0.5, movement=0.1, size=(100, 100)):
    filtering = DetectionFilter(confidence)
    matches = ObjectMatcher().match(filtering.apply(before), filtering.apply(after))
    return ChangeAnalyzer(movement).analyze(matches, size)


def test_all_four_statuses_and_json_output():
    before = [detection("b3", 3), detection("b0", 0), detection("b1", 1)]
    after = [
        detection("a2", 2), detection("a3", 3),
        detection("a0", 0, box=BoundingBox(90, 90, 92, 92)),
    ]
    records = run_pipeline(before, after)
    assert [(r["class_id"], r["status"]) for r in records] == [
        (0, "moved"), (1, "removed"), (2, "added"), (3, "unchanged"),
    ]
    assert json.loads(json.dumps(records, allow_nan=False)) == records


def test_duplicate_predictions_filtered_before_matcher():
    before = [detection("wrong", 0, 0.6, BoundingBox(90, 90, 92, 92)),
              detection("correct", 0, 0.95)]
    records = run_pipeline(before, [detection("a0", 0)])
    assert records[0]["status"] == "unchanged"
    assert records[0]["before_detection_id"] == "correct"


def test_low_confidence_after_produces_removed_prediction():
    records = run_pipeline([detection("b0", 0)], [detection("a0", 0, 0.1)])
    assert records[0]["status"] == "removed"
    assert records[0]["after_box"] is None


def test_low_confidence_before_produces_added_prediction():
    records = run_pipeline([detection("b0", 0, 0.1)], [detection("a0", 0)])
    assert records[0]["status"] == "added"


def test_all_filtered_or_empty_produces_no_events():
    assert run_pipeline([], []) == []
    assert run_pipeline([detection("b0", 0, 0.1)], [detection("a0", 0, 0.1)]) == []


def test_confidence_equality_and_movement_equality():
    before = [detection("b0", 0, 0.5, BoundingBox(0, 0, 1, 1))]
    after = [detection("a0", 0, 0.5, BoundingBox(3, 4, 4, 5))]
    record = run_pipeline(before, after, confidence=0.5, movement=0.5, size=(6, 8))[0]
    assert record["distance_norm"] == 0.5
    assert record["status"] == "unchanged"


def test_reordering_does_not_change_pipeline_output():
    before = [detection("b2", 2), detection("b0", 0, 0.7), detection("b0-high", 0)]
    after = [detection("a3", 3), detection("a0", 0)]
    assert run_pipeline(before, after) == run_pipeline(before[::-1], after[::-1])


def test_pipeline_does_not_mutate_original_lists():
    before, after = [detection("b0", 0)], [detection("a0", 0)]
    b0, a0 = before[0], after[0]
    run_pipeline(before, after)
    assert len(before) == len(after) == 1
    assert before[0] is b0 and after[0] is a0


def test_upper_image_bounds_are_validated_by_analyzer():
    invalid = detection("outside", 0, box=BoundingBox(0, 0, 200, 200))
    with pytest.raises(ValueError, match="image bounds"):
        run_pipeline([], [invalid], size=(100, 100))
