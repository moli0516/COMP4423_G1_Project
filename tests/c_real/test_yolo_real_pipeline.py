"""Real trained-model inference contract (injected YOLO stub only; not accuracy evidence)."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.real_settings import RealSettings
from deskcheck.detection.ultralytics_detector import UltralyticsDetector


class Boxes:
    def __init__(self, rows):
        self.xyxy = np.array([r[0] for r in rows], dtype=float).reshape(-1, 4)
        self.conf = np.array([r[1] for r in rows], dtype=float)
        self.cls = np.array([r[2] for r in rows], dtype=float)


class Model:
    names: ClassVar[dict[int, str]] = {
        0: "mouse",
        1: "bottle",
        2: "book",
        3: "cup",
        4: "keyboard",
        5: "scissors",
    }

    def __init__(self):
        self.calls = []

    def predict(self, source, **kwargs):
        assert source.dtype == np.uint8
        self.calls.append((source.copy(), kwargs))
        if tuple(source[0, 0]) == (0, 0, 255):
            rows = [([10, 10, 25, 25], 0.9, 1), ([40, 10, 55, 25], 0.8, 3)]
        else:
            rows = [([70, 10, 85, 25], 0.91, 1), ([40, 10, 55, 25], 0.8, 3)]
        return [type("Result", (), {"boxes": Boxes(rows), "orig_shape": source.shape[:2]})()]


class RealIntegration(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.weight = self.root / "best.pt"
        self.weight.write_bytes(b"NOT a real weight; test-only")
        self.config = self.root / "real.json"
        self.values = {
            "schema_version": "1.0",
            "mode": "real",
            "checkpoint_path": "best.pt",
            "checkpoint_sha256": hashlib.sha256(self.weight.read_bytes()).hexdigest(),
            "model_origin": "test-fixture-not-real",
            "confidence_threshold": 0.25,
            "movement_threshold": 0.05,
            "nms_iou_threshold": 0.5,
            "imgsz": 640,
            "rules_version": "test",
            "class_names": ["bottle", "cup", "keyboard", "mouse", "book", "scissors"],
        }
        self.config.write_text(json.dumps(self.values), encoding="utf-8")

    def test_real_backend_uses_image_pixels_maps_model_classes_and_reuses_b(self):
        settings = RealSettings.load(self.config)
        model = Model()
        detector = UltralyticsDetector(settings, model=model)
        service = ComparisonService(config=settings, detector=detector)
        before = np.zeros((120, 160, 3), dtype=np.uint8)
        after = before.copy()
        before[0, 0] = (255, 0, 0)
        after[0, 0] = (0, 0, 255)
        b, a = self.root / "b.png", self.root / "a.png"
        Image.fromarray(before).save(b)
        Image.fromarray(after).save(a)
        result = service.compare(b, a, "real-test")
        self.assertEqual(
            [(r["class_id"], r["status"]) for r in result["records"]],
            [(0, "moved"), (1, "unchanged")],
        )
        self.assertEqual(result["metadata"]["mode"], "real")
        self.assertEqual(result["metadata"]["checkpoint_sha256"], self.values["checkpoint_sha256"])
        self.assertEqual([x[0][0, 0].tolist() for x in model.calls], [[0, 0, 255], [255, 0, 0]])
        self.assertEqual(model.calls[0][1]["device"], "cpu")

    def test_missing_or_mismatched_weight_refuses_real_mode(self):
        self.weight.write_bytes(b"not matching checksum")
        with self.assertRaisesRegex(ValueError, "(?i)hash"):
            RealSettings.load(self.config)
        self.weight.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "(?i)checkpoint"):
            RealSettings.load(self.config)

    def test_malformed_real_config_and_missing_model_classes_fail(self):
        for key, invalid in (
            ("mode", "fake"),
            ("imgsz", 0),
            ("confidence_threshold", True),
            ("class_names", ["cup"]),
        ):
            original = self.values[key]
            self.values[key] = invalid
            self.config.write_text(json.dumps(self.values), encoding="utf-8")
            with self.subTest(key=key), self.assertRaises(ValueError):
                RealSettings.load(self.config)
            self.values[key] = original
        self.config.write_text(json.dumps(self.values), encoding="utf-8")
        model = Model()
        model.names = {0: "bottle", 1: "cup"}
        with self.assertRaisesRegex(ValueError, "(?i)class|missing"):
            UltralyticsDetector(RealSettings.load(self.config), model=model)

    def test_fake_app_config_refuses_real_config(self):
        from deskcheck.application.app_config import AppConfig

        with self.assertRaises(ValueError):
            AppConfig.load(self.config)
