"""EXIF orientation and shared frame identity across the application boundary."""

import builtins
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.result_serializer import ResultSerializer
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection
from deskcheck.domain.image_frame import ImageFrame

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
def test_mirrored_exif_to_shared_frame_keeps_rgb_and_original_boxes(orientation, monkeypatch):
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
            assert type(frame) is ImageFrame
            assert frame.source_path is None
            assert (frame.width, frame.height) == (width, height)
            assert frame.array.dtype == np.uint8
            # np.testing lazily reads package metadata; keep this no-I/O guard
            # focused on the already-loaded inference code and exact pixels.
            assert np.array_equal(frame.array, expected)
            seen.append(frame)
            return [prediction]

    service = ComparisonService(
        detector=StubDetector(),
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
    assert seen[0] is before and seen[1] is after
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


def test_corrupt_input_uses_pillow_decoder_after_detector_patches(monkeypatch):
    import sys
    from types import SimpleNamespace

    def process_wide_open(*args, **kwargs):
        pytest.fail("Loading a C upload should never invoke the detector's format installer")

    # Ultralytics 8.4.215 saves Pillow's opener, then replaces it process-wide.
    patches = SimpleNamespace(_image_open=Image.open, image_open=process_wide_open)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.patches", patches)
    monkeypatch.setattr(Image, "open", process_wide_open)
    with pytest.raises(ValueError, match="Cannot decode image input"):
        ImagePairLoader().load(b"broken", b"broken", "bad-upload")
    before, after = ImagePairLoader().load(png_bytes(PIXELS), png_bytes(PIXELS), "retry")
    np.testing.assert_array_equal(before.array, after.array)
