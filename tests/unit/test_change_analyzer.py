"""PR2 tests use synthetic thresholds, not approved dataset settings."""

import json
from math import hypot, inf, nan

import pytest

from deskcheck.changes.change_analyzer import ChangeAnalyzer
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection


def make_detection(detection_id="d0", class_id=0, *, box=None, confidence=0.9):
    return Detection(
        detection_id,
        class_id,
        confidence,
        box if box is not None else BoundingBox(0, 0, 2, 2),
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, nan, inf, -inf, None, "0.1", True])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Movement threshold"):
        ChangeAnalyzer(threshold)


@pytest.mark.parametrize("threshold", [0, 0.25, 1])
def test_accepts_threshold_range(threshold):
    assert ChangeAnalyzer(threshold).movement_threshold == threshold


def test_added_has_null_before_and_distance():
    after = make_detection("a0")
    records = ChangeAnalyzer(0.1).analyze([(None, after)], (10, 10))
    assert records == [{
        "class_id": 0,
        "status": "added",
        "before_detection_id": None,
        "after_detection_id": "a0",
        "before_box": None,
        "after_box": [0, 0, 2, 2],
        "distance_norm": None,
        "movement_threshold": 0.1,
        "indicates_change": True,
    }]


def test_removed_has_null_after_and_distance():
    before = make_detection("b0")
    record = ChangeAnalyzer(0.1).analyze([(before, None)], (10, 10))[0]
    assert record["status"] == "removed"
    assert record["before_detection_id"] == "b0"
    assert record["before_box"] == [0, 0, 2, 2]
    assert record["after_detection_id"] is None
    assert record["after_box"] is None
    assert record["distance_norm"] is None
    assert record["indicates_change"] is True


def test_identical_centers_are_unchanged_even_at_zero_threshold():
    before, after = make_detection("b0"), make_detection("a0")
    record = ChangeAnalyzer(0).analyze([(before, after)], (10, 10))[0]
    assert record["status"] == "unchanged"
    assert record["distance_norm"] == 0
    assert record["indicates_change"] is False


def test_center_displacement_is_normalized_by_original_diagonal():
    before = make_detection("b0")
    after = make_detection("a0", box=BoundingBox(3, 4, 5, 6))
    record = ChangeAnalyzer(0.1).analyze([(before, after)], (30, 40))[0]
    assert record["distance_norm"] == pytest.approx(5 / 50)
    assert record["status"] == "unchanged"


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.49, "moved"), (0.5, "unchanged"), (0.51, "unchanged")],
)
def test_exact_threshold_boundary_without_rounding(threshold, expected):
    # Image diagonal is exactly 10, and center displacement is exactly 5.
    before = make_detection("b0", box=BoundingBox(0, 0, 1, 1))
    after = make_detection("a0", box=BoundingBox(3, 4, 4, 5))
    record = ChangeAnalyzer(threshold).analyze([(before, after)], (6, 8))[0]
    assert record["distance_norm"] == 0.5
    assert record["status"] == expected
    assert record["indicates_change"] is (expected == "moved")


def test_far_matched_object_is_moved_not_added_or_removed():
    before = make_detection("b0")
    after = make_detection("a0", box=BoundingBox(90, 90, 92, 92))
    record = ChangeAnalyzer(0.1).analyze([(before, after)], (100, 100))[0]
    assert record["status"] == "moved"
    assert record["before_detection_id"] == "b0"
    assert record["after_detection_id"] == "a0"
    assert record["distance_norm"] == pytest.approx(hypot(90, 90) / hypot(100, 100))


def test_box_size_change_without_center_movement_is_unchanged():
    before = make_detection("b0", box=BoundingBox(2, 2, 4, 4))
    after = make_detection("a0", box=BoundingBox(1, 1, 5, 5))
    record = ChangeAnalyzer(0).analyze([(before, after)], (10, 10))[0]
    assert record["status"] == "unchanged"


def test_empty_matches_produce_no_records():
    assert ChangeAnalyzer(0.1).analyze([], (10, 10)) == []


@pytest.mark.parametrize(
    "size", [(0, 10), (10, 0), (-1, 10), (10, -1), (1.5, 10), (True, 10)]
)
def test_rejects_invalid_dimensions(size):
    with pytest.raises(ValueError, match="positive integers"):
        ChangeAnalyzer(0.1).analyze([], size)


@pytest.mark.parametrize("size", [None, (10,), (10, 10, 10), [10, 10]])
def test_rejects_invalid_image_size_shape(size):
    with pytest.raises(ValueError, match="tuple"):
        ChangeAnalyzer(0.1).analyze([], size)


@pytest.mark.parametrize("match", [(None, None), (), (None,), (None, None, None), "bad"])
def test_rejects_invalid_match_entries(match):
    with pytest.raises(ValueError):
        ChangeAnalyzer(0.1).analyze([match], (10, 10))


def test_rejects_non_detection_side():
    with pytest.raises(ValueError, match="Detection objects"):
        ChangeAnalyzer(0.1).analyze([({}, None)], (10, 10))


def test_rejects_mismatched_class_ids():
    with pytest.raises(ValueError, match="same class_id"):
        ChangeAnalyzer(0.1).analyze(
            [(make_detection("b0", 0), make_detection("a1", 1))], (10, 10)
        )


def test_rejects_duplicate_classes_across_matches():
    with pytest.raises(ValueError, match="Duplicate class_id 0"):
        ChangeAnalyzer(0.1).analyze(
            [(make_detection("b0"), None), (None, make_detection("a0"))], (10, 10)
        )


@pytest.mark.parametrize(
    "box", [BoundingBox(-1, 0, 1, 1), BoundingBox(0, 0, 20, 1),
            BoundingBox(0, 0, 0, 1), BoundingBox(nan, 0, 1, 1)],
)
@pytest.mark.parametrize("side", ["before", "after"])
def test_validates_boxes_on_each_present_side(box, side):
    detection = make_detection(box=box)
    match = (detection, None) if side == "before" else (None, detection)
    with pytest.raises(ValueError):
        ChangeAnalyzer(0.1).analyze([match], (10, 10))


def test_accepts_box_edges_at_image_boundary():
    detection = make_detection(box=BoundingBox(0, 0, 10, 10))
    assert ChangeAnalyzer(0.1).analyze([(None, detection)], (10, 10))[0]["status"] == "added"


def test_sorts_records_and_does_not_mutate_inputs_or_alias_boxes():
    before = make_detection("b3", 3)
    after = make_detection("a0", 0)
    matches = [(before, None), (None, after)]
    snapshot = matches.copy()
    records = ChangeAnalyzer(0.1).analyze(matches, (10, 10))
    assert [record["class_id"] for record in records] == [0, 3]
    assert matches == snapshot
    assert matches[0][0] is before
    records[1]["before_box"][0] = 999
    assert before.box.x1 == 0


def test_output_serializes_as_strict_json():
    matches = [(make_detection("b0"), make_detection("a0"))]
    records = ChangeAnalyzer(0.1).analyze(matches, (10, 10))
    assert json.loads(json.dumps(records, allow_nan=False)) == records


def test_analyzer_does_not_filter_confidence():
    after = make_detection("a0", confidence=0)
    assert ChangeAnalyzer(0.1).analyze([(None, after)], (10, 10))[0]["status"] == "added"


def test_no_state_leaks_between_calls():
    analyzer = ChangeAnalyzer(0.1)
    assert len(analyzer.analyze([(None, make_detection())], (10, 10))) == 1
    assert analyzer.analyze([], (10, 10)) == []
