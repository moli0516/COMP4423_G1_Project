"""P2: fake detector -> real filtering, matching and change analysis.

The composition below is test-only: C's ComparisonService is not exercised.
"""

import copy

import numpy as np
import pytest

from deskcheck.changes.change_analyzer import ChangeAnalyzer
from deskcheck.detection.detection_filter import DetectionFilter
from deskcheck.detection.object_detector import ObjectDetector
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection
from deskcheck.domain.image_frame import ImageFrame
from deskcheck.matching.object_matcher import ObjectMatcher

TEST_TAU = 0.1  # Test-only; not approval of the project movement threshold.
TEST_CONFIDENCE = 0.5


def _frame(marker):
    # Non-square frame: diagonal hypot(80, 60) == 100 pixels.
    return ImageFrame(
        array=np.full((60, 80, 3), marker, dtype=np.uint8),
        width=80, height=60, source_path=None,
    )


def _detection(identifier, class_id, x=5, y=5, confidence=0.9):
    return Detection(identifier, class_id, confidence, BoundingBox(x, y, x + 4, y + 4))


class ScriptedDetector(ObjectDetector):
    """Return independently supplied detections; inspect frames, never paths."""

    def __init__(self, before, after):
        self.responses = [before, after]
        self.calls = []

    def predict(self, image: ImageFrame) -> list[Detection]:
        assert isinstance(image, ImageFrame)
        assert image.source_path is None
        assert image.array.dtype == np.uint8
        assert image.array.shape == (image.height, image.width, 3)
        index = len(self.calls)
        assert index < 2, "Unexpected extra inference call"
        assert np.all(image.array == (11 if index == 0 else 22))
        self.calls.append(image)
        # Deliberately return original lists to expose downstream mutation.
        return self.responses[index]


def _run_pipeline(before, after, *, tau=TEST_TAU, reverse=False):
    before_frame, after_frame = _frame(11), _frame(22)
    inputs = [list(before), list(after)]
    if reverse:
        inputs = [list(reversed(items)) for items in inputs]
    snapshot = copy.deepcopy(inputs)
    arrays = [before_frame.array.copy(), after_frame.array.copy()]
    detector = ScriptedDetector(*inputs)
    detection_filter = DetectionFilter(confidence_threshold=TEST_CONFIDENCE)
    filtered_before = detection_filter.apply(detector.predict(before_frame))
    filtered_after = detection_filter.apply(detector.predict(after_frame))
    filtered_snapshot = (list(filtered_before), list(filtered_after))
    matches = ObjectMatcher().match(filtered_before, filtered_after)
    records = ChangeAnalyzer(movement_threshold=tau).analyze(
        matches, image_size=(before_frame.width, before_frame.height),
    )
    assert len(detector.calls) == 2
    assert detector.calls[0] is before_frame
    assert detector.calls[1] is after_frame
    assert inputs == snapshot
    assert (filtered_before, filtered_after) == filtered_snapshot
    np.testing.assert_array_equal(before_frame.array, arrays[0])
    np.testing.assert_array_equal(after_frame.array, arrays[1])
    return records


def test_p2_all_statuses_and_absent_both():
    before = [
        _detection("b2", 2), _detection("b1", 1), _detection("b0", 0),
    ]
    after = [
        _detection("a3", 3), _detection("a2", 2, x=25), _detection("a0", 0),
    ]
    records = _run_pipeline(before, after)
    assert [record["class_id"] for record in records] == [0, 1, 2, 3]
    assert [record["status"] for record in records] == [
        "unchanged", "removed", "moved", "added",
    ]
    assert [record["indicates_change"] for record in records] == [False, True, True, True]
    assert all(record["movement_threshold"] == TEST_TAU for record in records)
    assert records[0]["distance_norm"] == 0
    assert records[2]["distance_norm"] == pytest.approx(0.2)
    assert records[1]["distance_norm"] is None
    assert records[3]["distance_norm"] is None
    assert records[0]["before_box"] == [5, 5, 9, 9]
    assert records[2]["after_box"] == [25, 5, 29, 9]
    assert records[1]["after_box"] is None
    assert records[1]["after_detection_id"] is None
    assert records[3]["before_box"] is None
    assert records[3]["before_detection_id"] is None
    assert records == _run_pipeline(before, after, reverse=True)


@pytest.mark.parametrize(
    ("dx", "expected"), [(9, "unchanged"), (10, "unchanged"), (11, "moved")],
)
def test_p2_movement_boundary(dx, expected):
    records = _run_pipeline(
        [_detection("b", 0)], [_detection("a", 0, x=5 + dx)],
    )
    assert len(records) == 1
    assert records[0]["distance_norm"] == pytest.approx(dx / 100)
    assert records[0]["status"] == expected
    assert records[0]["indicates_change"] == (expected == "moved")


def test_p2_confidence_filter_changes_presence_and_keeps_equality():
    before = [
        _detection("b0", 0, confidence=0.49),
        _detection("b1", 1, confidence=0.5),
        _detection("b2", 2, confidence=0.5),
        _detection("b3", 3, confidence=0.49),
    ]
    after = [
        _detection("a0", 0, confidence=0.5),
        _detection("a1", 1, confidence=0.49),
        _detection("a2", 2, confidence=0.5),
        _detection("a3", 3, confidence=0.49),
    ]
    records = _run_pipeline(before, after)
    assert [(r["class_id"], r["status"]) for r in records] == [
        (0, "added"), (1, "removed"), (2, "unchanged"),
    ]
    assert records[0]["before_detection_id"] is None
    assert records[0]["after_detection_id"] == "a0"
    assert records[1]["before_detection_id"] == "b1"
    assert records[2]["before_detection_id"] == "b2"


def test_p2_highest_confidence_duplicate_is_selected_before_matching():
    before = [
        _detection("low", 0, x=5, confidence=0.8),
        _detection("high", 0, x=25, confidence=0.9),
    ]
    after = [_detection("after", 0, x=25)]
    records = _run_pipeline(before, after)
    assert len(records) == 1
    assert records[0]["before_detection_id"] == "high"
    assert records[0]["before_box"] == [25, 5, 29, 9]
    assert records[0]["status"] == "unchanged"
    assert records == _run_pipeline(before, after, reverse=True)


def test_p2_equal_confidence_uses_geometry_then_identifier():
    before = [
        _detection("later-geometry", 0, x=25),
        _detection("z", 0, x=5), _detection("a", 0, x=5),
    ]
    after = [_detection("after", 0, x=5)]
    records = _run_pipeline(before, after)
    assert records[0]["before_detection_id"] == "a"
    assert records[0]["status"] == "unchanged"
    assert records == _run_pipeline(before, after, reverse=True)


def test_p2_far_same_class_is_moved_not_added_and_removed():
    records = _run_pipeline(
        [_detection("before", 0, x=1, y=1)],
        [_detection("after", 0, x=70, y=50)],
    )
    assert len(records) == 1
    assert records[0]["class_id"] == 0
    assert records[0]["status"] == "moved"
    assert records[0]["before_detection_id"] == "before"
    assert records[0]["after_detection_id"] == "after"


def test_p2_empty_frames_produce_no_records():
    assert _run_pipeline([], []) == []


def test_p2_different_classes_are_not_matched_by_overlap():
    records = _run_pipeline([_detection("before", 0)], [_detection("after", 1)])
    assert [(r["class_id"], r["status"]) for r in records] == [
        (0, "removed"), (1, "added"),
    ]


def test_p2_out_of_frame_detection_is_not_silently_accepted():
    with pytest.raises(ValueError):
        _run_pipeline([_detection("bad", 0, x=79)], [])
