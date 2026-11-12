"""C result exposes post-B-filter application diagnostics, not detector AP input.

These are application predictions, never labels or a measured AP score.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.demo_pair import make_demo_pair
from deskcheck.application.image_frame import ImageFrame
from deskcheck.application.result_serializer import ResultSerializer
from deskcheck.detection.fake_color_detector import FakeColorDetector
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection

ROOT = Path(__file__).resolve().parents[2]


class RetainedDetectionsContractTests(unittest.TestCase):
    def config(self):
        return AppConfig.load(ROOT / "configs/fake_app.json")

    def test_real_b_filtered_predictions_are_saved_with_original_boxes(self):
        before, after = make_demo_pair()
        service = ComparisonService(detector=FakeColorDetector(), config=self.config())
        result = service.compare(before, after, "pair-for-A")
        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual([r["class_id"] for r in result["before_detections"]], [0, 1, 2])
        self.assertEqual([r["class_id"] for r in result["after_detections"]], [0, 2, 3])
        first = result["before_detections"][0]
        self.assertEqual(
            first,
            {
                "detection_id": "fake-0",
                "class_id": 0,
                "confidence": 1.0,
                "bbox_xyxy": [10.0, 10.0, 25.0, 25.0],
            },
        )
        self.assertEqual(result["metadata"]["prediction_scope"], "post_B_filter")
        encoded = json.loads(ResultSerializer().to_json(result))
        self.assertEqual(encoded["before_detections"], result["before_detections"])
        self.assertEqual(encoded["after_detections"], result["after_detections"])

    def test_export_uses_B_selected_results_not_raw_or_label_side_channel(self):
        class ExtraRawDetections:
            def predict(self, frame):
                return [
                    Detection("low", 0, 0.51, BoundingBox(1, 1, 5, 5)),
                    Detection("best", 0, 0.9, BoundingBox(3, 3, 7, 7)),
                    Detection("below", 1, 0.1, BoundingBox(4, 4, 8, 8)),
                ]

        service = ComparisonService(detector=ExtraRawDetections(), config=self.config())
        frame = ImageFrame(np.zeros((15, 15, 3), dtype=np.uint8))
        result = service.compare_frames(frame, frame, pair_id="no-gold")
        self.assertEqual(
            result["before_detections"],
            [
                {
                    "detection_id": "best",
                    "class_id": 0,
                    "confidence": 0.9,
                    "bbox_xyxy": [3.0, 3.0, 7.0, 7.0],
                }
            ],
        )
        self.assertEqual(result["after_detections"], result["before_detections"])
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["status"], "unchanged")

    def test_serializer_rejects_invalid_exposed_detector_geometry(self):
        before, after = make_demo_pair()
        result = ComparisonService(detector=FakeColorDetector(), config=self.config()).compare(
            before, after, "invalid-detect"
        )
        for bad in (
            {"bbox_xyxy": [0, 0, 9000, 25]},
            {"confidence": float("nan")},
            {"class_id": 99},
        ):
            with self.subTest(bad=bad):
                changed = {
                    **result,
                    "before_detections": [{**result["before_detections"][0], **bad}],
                }
                with self.assertRaises(ValueError):
                    ResultSerializer().to_json(changed)
