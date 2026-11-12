"""STUB_ONLY: C's temporary frame seam, not A's unmerged production type.

Exercise missing geometry and failure boundaries without importing PR #8,
opening a model, or replacing the existing C/B implementations.
"""

import builtins
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.frame_adapter import FrameDetectorAdapter
from deskcheck.application.image_frame import ImageFrame
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.result_serializer import ResultSerializer
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection

ROOT = Path(__file__).resolve().parents[2]
PIXELS = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)


def png_bytes(pixels, orientation=None):
    output = BytesIO()
    with Image.fromarray(pixels) as image:
        exif = Image.Exif()
        if orientation is not None:
            exif[274] = orientation
        image.save(output, format="PNG", exif=exif)
    return output.getvalue()


@pytest.mark.parametrize("orientation", [2, 4, 5, 7])
def test_mirrored_exif_to_stub_adapter_keeps_rgb_and_original_boxes(orientation, monkeypatch):
    # Independent NumPy oracles; lossless encoding makes every RGB byte checkable.
    expected = {
        2: PIXELS[:, ::-1, :],
        4: PIXELS[::-1, :, :],
        5: PIXELS.transpose(1, 0, 2),
        7: PIXELS.transpose(1, 0, 2)[::-1, ::-1, :],
    }[orientation]
    before, after = ImagePairLoader().load(
        png_bytes(PIXELS, orientation), png_bytes(expected), "mirror"
    )
    height, width, _ = expected.shape
    prediction = Detection("edge", 5, 0.9, BoundingBox(0, 0, width, height))
    seen = []

    class StubDetector:
        def predict(self, frame):
            assert isinstance(frame, SimpleNamespace)
            assert set(vars(frame)) == {"array", "width", "height", "source_path"}
            assert frame.source_path is None
            assert (frame.width, frame.height) == (width, height)
            assert frame.array.dtype == np.uint8
            # np.testing lazily reads package metadata; keep this no-I/O guard
            # focused on the already-loaded inference code and exact pixels.
            assert np.array_equal(frame.array, expected)
            assert not np.shares_memory(frame.array, before.array)
            assert not np.shares_memory(frame.array, after.array)
            seen.append(frame)
            return [prediction]

    service = ComparisonService(
        detector=FrameDetectorAdapter(StubDetector(), SimpleNamespace),
        config=AppConfig.load(ROOT / "configs/fake_app.json"),
    )

    def no_file_reopening(*args, **kwargs):
        raise AssertionError("The decoded-frame detector path must not reopen files")

    # Configuration/decoding are complete. Inference receives only copied pixels.
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", no_file_reopening)
        guard.setattr(Path, "open", no_file_reopening)
        guard.setattr(Image, "open", no_file_reopening)
        result = ResultSerializer().snapshot(
            service.compare_frames(before, after, pair_id="mirror")
        )

    assert len(seen) == 2
    assert not np.shares_memory(seen[0].array, seen[1].array)
    for frame in (before, after):
        np.testing.assert_array_equal(frame.array, expected)
        assert not frame.array.flags.writeable
    assert result["image_size"] == [width, height]
    for side in ("before", "after"):
        assert result[f"{side}_detections"] == [{
            "detection_id": "edge", "class_id": 5, "confidence": 0.9,
            "bbox_xyxy": [0.0, 0.0, float(width), float(height)],
        }]
        assert result["records"][0][f"{side}_box"] == [0, 0, width, height]
    assert result["records"][0]["class_id"] == 5
    assert result["records"][0]["status"] == "unchanged"
    assert result["metadata"]["mode"] == "fake"
    assert result["metadata"]["checkpoint_sha256"] is None


@pytest.mark.parametrize("invalid_field", ["width", "height", "source_path", "rgb", "dtype"])
def test_factory_must_preserve_canonical_frame_before_detector_call(invalid_field):
    frame = ImageFrame(PIXELS)
    detector = Mock()

    def invalid_factory(**fields):
        if invalid_field in ("width", "height"):
            fields[invalid_field] += 1
        elif invalid_field == "source_path":
            fields["source_path"] = Path("must-not-be-opened.png")
        elif invalid_field == "rgb":
            fields["array"] = fields["array"][:, :, ::-1]
        else:
            fields["array"] = fields["array"].astype(np.float32)
        return SimpleNamespace(**fields)

    with pytest.raises(ValueError, match="Frame factory changed canonical"):
        FrameDetectorAdapter(detector, invalid_factory).predict(frame)
    detector.predict.assert_not_called()
    np.testing.assert_array_equal(frame.array, PIXELS)


@pytest.mark.parametrize("field", ["width", "height", "source_path"])
def test_detector_cannot_change_frame_metadata(field):
    frame = ImageFrame(PIXELS)

    class MetadataMutatingStub:
        def predict(self, converted):
            setattr(converted, field, "changed")
            return []

    with pytest.raises(ValueError, match="Detector mutated canonical"):
        FrameDetectorAdapter(MetadataMutatingStub(), SimpleNamespace).predict(frame)
    assert (frame.width, frame.height) == (5, 3)
    np.testing.assert_array_equal(frame.array, PIXELS)


@pytest.mark.parametrize("bad_output", [None, (), [SimpleNamespace(
    detection_id="duck-typed", class_id=5, confidence=0.9, box=BoundingBox(0, 0, 5, 3)
)]], ids=["missing-list", "tuple", "foreign-detection-type"])
def test_adapter_requires_list_of_actual_b_detection_objects(bad_output):
    detector = Mock()
    detector.predict.return_value = bad_output
    frame = ImageFrame(PIXELS)
    with pytest.raises(TypeError, match="list of shared Detection"):
        FrameDetectorAdapter(detector, SimpleNamespace).predict(frame)
    np.testing.assert_array_equal(frame.array, PIXELS)


def test_adapter_preserves_actual_b_detection_identity():
    prediction = Detection("canonical", 5, 0.9, BoundingBox(0, 0, 5, 3))
    detector = Mock()
    detector.predict.return_value = [prediction]
    actual = FrameDetectorAdapter(detector, SimpleNamespace).predict(ImageFrame(PIXELS))
    assert actual[0] is prediction
