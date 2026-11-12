"""Synthetic transport/presentation regressions; no detector accuracy evidence."""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import ImageDraw

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.demo_pair import make_demo_pair
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.presentation import comparison_rows
from deskcheck.application.result_renderer import ResultRenderer
from deskcheck.application.result_serializer import ResultSerializer
from deskcheck.detection.fake_color_detector import FakeColorDetector

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def result():
    return ComparisonService(
        detector=FakeColorDetector(), config=AppConfig.load(ROOT / "configs/fake_app.json")
    ).compare(*make_demo_pair(), pair_id="evidence")


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("change", ["id", "box", "class", "missing", "extra"])
def test_contradictory_retained_evidence_cannot_be_exported(result, side, change):
    detections = result[f"{side}_detections"]
    if change == "id":
        detections[0]["detection_id"] = "different-detection"
    elif change == "box":
        detections[0]["bbox_xyxy"][0] += 1
    elif change == "class":
        detections[0]["class_id"] = 5
    elif change == "missing":
        detections.pop()
    else:
        detections.append(
            {"detection_id": "extra", "class_id": 5, "confidence": 0.8, "bbox_xyxy": [1, 1, 4, 4]}
        )
    for export in (ResultSerializer().to_json, ResultSerializer().to_csv):
        with pytest.raises(ValueError, match="(?i)retained.*(event|record)"):
            export(result)


def test_absent_side_id_must_also_be_null(result):
    added = next(r for r in result["records"] if r["status"] == "added")
    added["before_detection_id"] = "ghost"
    with pytest.raises(ValueError, match="(?i)retained.*(event|record)"):
        ResultSerializer().snapshot(result)


def test_legacy_event_only_and_explicit_empty_scene_remain_supported(result):
    legacy = {k: v for k, v in result.items() if k not in ("before_detections", "after_detections")}
    assert ResultSerializer().snapshot(legacy) == legacy
    empty = {**result, "records": [], "before_detections": [], "after_detections": []}
    assert ResultSerializer().snapshot(empty) == empty
    assert comparison_rows(empty) == []


def test_confidence_and_correspondence_are_visible_without_changing_result(result):
    saved = copy.deepcopy(result)
    rows = comparison_rows(result)
    bottle = next(r for r in rows if r["Object"] == "bottle")
    assert bottle["Before confidence"] == 1.0
    assert bottle["After confidence"] == 1.0
    removed = next(r for r in rows if r["Status"] == "removed")
    assert removed["After confidence"] is None
    frames = ImagePairLoader().load(*make_demo_pair(), pair_id="evidence")
    drawn = []
    original = ImageDraw.ImageDraw.text

    def capture(draw, xy, text, *args, **kwargs):
        drawn.append(text)
        return original(draw, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", capture):
        images = ResultRenderer().render(*frames, result)
    assert drawn.count("object-0") == 2
    assert any("conf" in text and "1.000" in text for text in drawn)
    assert images[0].tobytes() != frames[0].array.tobytes()
    assert result == saved
