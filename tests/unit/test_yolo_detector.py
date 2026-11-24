"""Unit tests for YoloDetector using a mocked ultralytics backend.

The real model is exercised separately in the integration test
(tests/integration/test_detector_to_b.py) once a checkpoint is present.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deskcheck.detection.yolo_detector import YoloConfig, YoloDetector
from deskcheck.domain.image_frame import ImageFrame

COCO_NAMES = {
    39: "bottle",
    41: "cup",
    64: "mouse",
    66: "keyboard",
    73: "book",
    76: "scissors",
    0: "person",  # will be dropped by canonical map
}


class _FakeBox:
    def __init__(self, cls: float, conf: float, xyxy: list[float]):
        self.cls = [MagicMock(item=lambda v=cls: float(v))]
        self.conf = [MagicMock(item=lambda v=conf: float(v))]
        self.xyxy = [MagicMock(tolist=lambda xy=xyxy: list(xy))]

    def __len__(self) -> int:
        return 1


class _FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = COCO_NAMES
        self.orig_shape = (480, 640)


def _make_frame(w: int = 640, h: int = 480) -> ImageFrame:
    return ImageFrame(
        array=np.zeros((h, w, 3), dtype=np.uint8),
        width=w,
        height=h,
        source_path=None,
    )


def _make_detector(tmp_path: Path, **overrides) -> YoloDetector:
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    cfg = YoloConfig(checkpoint_path=ckpt, **overrides)
    return YoloDetector(cfg)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    cfg = YoloConfig(checkpoint_path=tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        YoloDetector(cfg)


def test_invalid_confidence_raises(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(ValueError, match="confidence_threshold"):
        YoloDetector(YoloConfig(checkpoint_path=ckpt, confidence_threshold=1.5))


def test_invalid_nms_iou_raises(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(ValueError, match="nms_iou_threshold"):
        YoloDetector(YoloConfig(checkpoint_path=ckpt, nms_iou_threshold=-0.1))


def test_predict_rejects_non_image_frame(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    with pytest.raises(TypeError, match="ImageFrame"):
        det.predict("not_a_frame")  # type: ignore[arg-type]


def test_predict_maps_canonical_classes(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    boxes = [
        _FakeBox(cls=39, conf=0.9, xyxy=[10.0, 20.0, 100.0, 200.0]),  # bottle -> 0
        _FakeBox(cls=41, conf=0.8, xyxy=[200.0, 50.0, 300.0, 150.0]), # cup -> 1
        _FakeBox(cls=66, conf=0.7, xyxy=[300.0, 400.0, 600.0, 460.0]),# keyboard -> 2
    ]
    fake = _FakeResults(boxes)
    fake_model = MagicMock()
    fake_model.names = COCO_NAMES
    fake_model.predict.return_value = [fake]

    with patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)):
        dets = det.predict(_make_frame())

    assert [d.class_id for d in dets] == [0, 1, 2]
    assert all(isinstance(d.confidence, float) for d in dets)
    assert dets[0].box.x1 == 10.0


def test_predict_drops_unsupported_classes(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    boxes = [
        _FakeBox(cls=0, conf=0.99, xyxy=[0, 0, 50, 50]),     # person -> dropped
        _FakeBox(cls=39, conf=0.9, xyxy=[10, 10, 100, 100]), # bottle -> kept
    ]
    fake = _FakeResults(boxes)
    fake_model = MagicMock()
    fake_model.names = COCO_NAMES
    fake_model.predict.return_value = [fake]

    with patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)):
        dets = det.predict(_make_frame())

    assert len(dets) == 1
    assert dets[0].class_id == 0


@pytest.mark.parametrize("class_id", [0.9, -0.9, float("nan"), float("inf"), float("-inf")])
def test_predict_rejects_malformed_backend_class_id(tmp_path: Path, class_id: float) -> None:
    det = _make_detector(tmp_path)
    model = MagicMock()
    model.predict.return_value = [_FakeResults([_FakeBox(class_id, 0.9, [0, 0, 50, 50])])]

    with (
        patch.object(det, "_load_model", return_value=(model, {0: "bottle"})),
        pytest.raises(ValueError, match="class ID"),
    ):
        det.predict(_make_frame())


def test_predict_valid_empty_returns_empty_list(tmp_path: Path) -> None:
    """Valid empty: Results with a real Boxes object of length 0."""
    class EmptyBoxes:
        def __len__(self):
            return 0

    det = _make_detector(tmp_path)
    fake_model = MagicMock()
    fake_model.names = COCO_NAMES
    fake_model.predict.return_value = [_FakeResults(EmptyBoxes())]

    with patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)):
        assert det.predict(_make_frame()) == []


def test_predict_raises_on_none_results(tmp_path: Path) -> None:
    """Malformed: results is None, not a list. TypeError (not RuntimeError)
    because the return value has the wrong type."""
    det = _make_detector(tmp_path)
    fake_model = MagicMock()
    fake_model.names = COCO_NAMES
    fake_model.predict.return_value = None

    with (
        patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)),
        pytest.raises(TypeError, match="expected list of Results"),
    ):
        det.predict(_make_frame())


def test_predict_raises_on_missing_boxes_attr(tmp_path: Path) -> None:
    """Malformed: Results has no .boxes attribute."""
    det = _make_detector(tmp_path)
    fake_model = MagicMock()
    fake_model.names = COCO_NAMES
    result_no_boxes = object()
    fake_model.predict.return_value = [result_no_boxes]

    with (
        patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)),
        pytest.raises(RuntimeError, match="missing .boxes. attribute"),
    ):
        det.predict(_make_frame())


def test_predict_raises_on_boxes_none(tmp_path: Path) -> None:
    """Malformed: Result.boxes is None (distinct from valid empty)."""
    det = _make_detector(tmp_path)
    fake_model = MagicMock()
    fake_model.names = COCO_NAMES
    result = MagicMock()
    result.boxes = None
    fake_model.predict.return_value = [result]

    with (
        patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)),
        pytest.raises(RuntimeError, match="boxes is None"),
    ):
        det.predict(_make_frame())


def test_predict_sorts_by_class_then_confidence(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    boxes = [
        _FakeBox(cls=66, conf=0.5, xyxy=[0, 0, 10, 10]),    # keyboard class 2
        _FakeBox(cls=39, conf=0.9, xyxy=[0, 0, 10, 10]),    # bottle class 0
        _FakeBox(cls=39, conf=0.7, xyxy=[0, 0, 10, 10]),    # bottle class 0 lower conf
    ]
    fake = _FakeResults(boxes)
    fake_model = MagicMock()
    fake_model.names = COCO_NAMES
    fake_model.predict.return_value = [fake]

    with patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)):
        dets = det.predict(_make_frame())

    # class 0 (bottle) first, higher conf first
    assert [d.class_id for d in dets] == [0, 0, 2]
    assert dets[0].confidence > dets[1].confidence


def test_predict_does_not_mutate_input_array(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    # A caller can supply canonical RGB as a view of a contiguous BGR buffer.
    backing = np.broadcast_to(np.array([11, 22, 33], dtype=np.uint8), (4, 5, 3)).copy()
    frame = ImageFrame(backing[:, :, ::-1], width=5, height=4)
    original = frame.array.copy()

    def predict(*, source, **kwargs):
        assert source.flags.c_contiguous
        assert np.array_equal(source, backing)
        source.fill(0)
        result = _FakeResults([])
        result.orig_shape = (frame.height, frame.width)
        return [result]

    fake_model = MagicMock()
    fake_model.predict.side_effect = predict
    with patch.object(det, "_load_model", return_value=(fake_model, COCO_NAMES)):
        det.predict(frame)

    assert np.array_equal(frame.array, original)


@pytest.mark.parametrize("wrong", [[], [_FakeResults([]), _FakeResults([])]])
def test_unexpected_results_cannot_be_published_as_an_empty_scene(tmp_path, wrong):
    det = _make_detector(tmp_path)
    model = MagicMock()
    model.predict.return_value = wrong
    with (
        patch.object(det, "_load_model", return_value=(model, COCO_NAMES)),
        pytest.raises(RuntimeError, match="expected exactly 1"),
    ):
        det.predict(_make_frame())


def test_rejects_detector_geometry_from_another_orientation(tmp_path):
    det = _make_detector(tmp_path)
    model = MagicMock()
    result = _FakeResults([])
    result.orig_shape = (640, 480)
    model.predict.return_value = [result]
    with (
        patch.object(det, "_load_model", return_value=(model, COCO_NAMES)),
        pytest.raises(ValueError, match="geometry"),
    ):
        det.predict(_make_frame())
