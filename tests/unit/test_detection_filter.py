"""Synthetic confidence settings are test fixtures, not deployment choices."""

from itertools import permutations
from math import inf, nan

import pytest

from deskcheck.detection.detection_filter import DetectionFilter
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection


def make_detection(detection_id="d0", class_id=0, confidence=0.9, box=None):
    return Detection(
        detection_id, class_id, confidence,
        box if box is not None else BoundingBox(0, 0, 2, 2),
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, nan, inf, -inf, None, "0.5", True])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Confidence threshold"):
        DetectionFilter(threshold)


@pytest.mark.parametrize("threshold", [0, 0.5, 1])
def test_valid_threshold(threshold):
    assert DetectionFilter(threshold).confidence_threshold == threshold


def test_empty_input():
    assert DetectionFilter(0.5).apply([]) == []


def test_confidence_boundary_is_inclusive():
    below = make_detection("below", 0, 0.49)
    equal = make_detection("equal", 1, 0.5)
    above = make_detection("above", 2, 0.51)
    assert DetectionFilter(0.5).apply([below, equal, above]) == [equal, above]


def test_all_below_threshold():
    assert DetectionFilter(0.5).apply([make_detection(confidence=0.1)]) == []


@pytest.mark.parametrize("threshold, confidence", [(0, 0), (1, 1)])
def test_zero_and_one_boundaries(threshold, confidence):
    detection = make_detection(confidence=confidence)
    assert DetectionFilter(threshold).apply([detection]) == [detection]


def test_highest_confidence_wins_even_without_box_overlap():
    low = make_detection("low", confidence=0.7)
    high = make_detection("high", confidence=0.9, box=BoundingBox(90, 90, 92, 92))
    result = DetectionFilter(0.5).apply([low, high])
    assert len(result) == 1
    assert result[0] is high


@pytest.mark.parametrize(
    "worse_box", [BoundingBox(1, 0, 3, 2), BoundingBox(0, 1, 2, 3),
                  BoundingBox(0, 0, 3, 2), BoundingBox(0, 0, 2, 3)],
)
def test_equal_confidence_tie_uses_xyxy_order(worse_box):
    winner = make_detection("z", box=BoundingBox(0, 0, 2, 2))
    loser = make_detection("a", box=worse_box)
    for order in ([loser, winner], [winner, loser]):
        assert DetectionFilter(0.5).apply(order)[0] is winner


def test_equal_confidence_and_boxes_tie_uses_id():
    winner, loser = make_detection("a"), make_detection("z")
    assert DetectionFilter(0.5).apply([loser, winner])[0] is winner


def test_fully_identical_keys_preserve_first_object():
    first, second = make_detection(), make_detection()
    assert first is not second
    assert DetectionFilter(0.5).apply([first, second])[0] is first


def test_exact_repeated_object_is_collapsed():
    detection = make_detection()
    assert DetectionFilter(0.5).apply([detection, detection]) == [detection]


def test_different_classes_survive_identical_geometry():
    a, b = make_detection("a", 0), make_detection("b", 1)
    assert DetectionFilter(0.5).apply([b, a]) == [a, b]


def test_permutations_produce_same_selection_and_sorted_output():
    a = make_detection("a", 0, 0.8)
    b = make_detection("b", 0, 0.9)
    c = make_detection("c", 3, 0.9)
    for order in permutations([a, b, c]):
        assert DetectionFilter(0.5).apply(list(order)) == [b, c]


def test_does_not_mutate_input_and_preserves_selected_object_identity():
    high = make_detection("high", 3)
    low = make_detection("low", 0, 0.1)
    original = [high, low]
    result = DetectionFilter(0.5).apply(original)
    assert len(original) == 2
    assert original[0] is high and original[1] is low
    assert result is not original and result[0] is high


def test_calls_do_not_share_selection_state():
    detection_filter = DetectionFilter(0.5)
    assert len(detection_filter.apply([make_detection()])) == 1
    assert detection_filter.apply([]) == []


@pytest.mark.parametrize("value", [None, {}, "detection"])
def test_invalid_detection_types(value):
    with pytest.raises(TypeError, match="must be a Detection"):
        DetectionFilter(0.5).apply([value])


@pytest.mark.parametrize(
    "box", [BoundingBox(nan, 0, 2, 2), BoundingBox(0, 0, inf, 2),
            BoundingBox(-1, 0, 2, 2), BoundingBox(0, 0, 0, 2)],
)
def test_invalid_box_is_rejected_even_below_threshold(box):
    with pytest.raises(ValueError):
        DetectionFilter(0.5).apply([make_detection(confidence=0.1, box=box)])


def test_fractional_class_id_is_rejected():
    with pytest.raises(ValueError, match="nonnegative integer"):
        DetectionFilter(0.5).apply([make_detection(class_id=0.5)])


def test_nonstring_detection_id_is_rejected():
    with pytest.raises(TypeError, match="must be a string"):
        DetectionFilter(0.5).apply([make_detection(detection_id=123)])
