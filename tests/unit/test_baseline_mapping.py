"""Tests for deriving the baseline class mapping from a checkpoint's names."""

from __future__ import annotations

import numpy as np
import pytest

from deskcheck.detection.object_detector import ObjectDetector
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection
from deskcheck.domain.image_frame import ImageFrame
from deskcheck.evaluation.baseline_mapping import (
    BaselineClassDetector,
    map_baseline_names,
)

PROJECT_NAMES = ("bottle", "cup", "keyboard", "mouse", "book", "scissors")


def _names(positions: dict[str, int], total: int = 80) -> dict[int, str]:
    """Build a contiguous names dict with project classes at the given indices."""
    names = {index: f"filler_{index}" for index in range(total)}
    for name, index in positions.items():
        names[index] = name
    return names


def _index_of(mapping, project_id: int) -> int:
    return next(
        index for index, value in mapping.class_map.items() if value == project_id
    )


def test_maps_project_classes_at_arbitrary_indices() -> None:
    positions = {"bottle": 3, "cup": 7, "keyboard": 11, "mouse": 42, "book": 63,
                 "scissors": 79}

    mapping = map_baseline_names(_names(positions))

    assert dict(mapping.class_map) == {3: 0, 7: 1, 11: 2, 42: 3, 63: 4, 79: 5}
    assert len(mapping.dropped) == 74
    assert 3 not in dict(mapping.dropped)
    assert mapping.remap(42) == 3
    assert mapping.remap(0) is None


def test_permuting_indices_changes_the_mapping() -> None:
    first = map_baseline_names(_names(
        {"bottle": 1, "cup": 2, "keyboard": 3, "mouse": 4, "book": 5, "scissors": 6}
    ))
    second = map_baseline_names(_names(
        {"bottle": 6, "cup": 5, "keyboard": 4, "mouse": 3, "book": 2, "scissors": 1}
    ))

    assert dict(first.class_map) != dict(second.class_map)
    for project_id, name in enumerate(PROJECT_NAMES):
        assert first.baseline_names[_index_of(first, project_id)] == name
        assert second.baseline_names[_index_of(second, project_id)] == name
    assert _index_of(first, 0) == 1 and _index_of(second, 0) == 6


def test_missing_class_is_rejected() -> None:
    names = _names({name: index for index, name in enumerate(PROJECT_NAMES)})
    del names[4]
    names[4] = "filler_4"

    with pytest.raises(ValueError, match="missing project class 'book'"):
        map_baseline_names(names)


def test_differently_spelled_class_is_rejected_with_the_spelling() -> None:
    positions = {name: index for index, name in enumerate(PROJECT_NAMES)}
    positions["Books"] = positions.pop("book")

    with pytest.raises(ValueError, match=r"missing project class 'book'.*Books"):
        map_baseline_names(_names(positions))


def test_duplicated_class_is_rejected() -> None:
    positions = {name: index for index, name in enumerate(PROJECT_NAMES)}
    names = _names(positions)
    names[9] = "book"

    with pytest.raises(ValueError, match=r"'book' is duplicated at indices \[4, 9\]"):
        map_baseline_names(names)


def test_ambiguous_class_is_rejected() -> None:
    positions = {name: index for index, name in enumerate(PROJECT_NAMES)}
    names = _names(positions)
    names[9] = "Book"

    with pytest.raises(ValueError, match=r"'book' is ambiguous.*\['Book'\]"):
        map_baseline_names(names)


def test_non_contiguous_indices_are_rejected() -> None:
    names = _names({name: index for index, name in enumerate(PROJECT_NAMES)})
    del names[10]
    names[11] = "shifted"

    with pytest.raises(ValueError, match="contiguous from 0"):
        map_baseline_names(names)


@pytest.mark.parametrize(
    "names",
    [{}, {"bottle": "x"}, {0: ""}, {0: "person", 1: 5}],
)
def test_malformed_names_are_rejected(names: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        map_baseline_names(names)


class _StubDetector(ObjectDetector):
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.seen: list[ImageFrame] = []

    def predict(self, image: ImageFrame) -> list[Detection]:
        self.seen.append(image)
        return list(self.detections)


def _detection(detection_id: str, class_id: int, confidence: float) -> Detection:
    return Detection(
        detection_id=detection_id,
        class_id=class_id,
        confidence=confidence,
        box=BoundingBox(0.0, 0.0, 10.0, 10.0),
    )


def test_detector_wrapper_remaps_and_drops_without_mutation() -> None:
    mapping = map_baseline_names(_names(
        {"bottle": 3, "cup": 7, "keyboard": 11, "mouse": 42, "book": 63, "scissors": 79}
    ))
    inner = _StubDetector([
        _detection("a", 3, 0.90),
        _detection("b", 79, 0.50),
        _detection("c", 0, 0.99),
        _detection("d", 42, 0.70),
    ])
    detector = BaselineClassDetector(detector=inner, mapping=mapping)
    frame = ImageFrame(np.zeros((4, 4, 3), dtype=np.uint8), width=4, height=4)

    result = detector.predict(frame)

    assert [(item.detection_id, item.class_id) for item in result] == [
        ("a", 0), ("d", 3), ("b", 5),
    ]
    assert [item.class_id for item in inner.detections] == [3, 79, 0, 42]
    assert inner.seen == [frame]


def test_detector_wrapper_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        BaselineClassDetector(detector=_StubDetector([]), mapping={"bottle": 0})
