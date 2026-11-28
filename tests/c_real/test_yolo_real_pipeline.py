"""Real trained-model inference contract (injected YOLO stub only; not accuracy evidence)."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.real_settings import RealSettings


class Box:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.array([xyxy], dtype=float)
        self.conf = np.array([confidence])
        self.cls = np.array([class_id])


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
        self.constructor = Mock(return_value=self)

    def installed(self):
        """Substitute only the optional YOLO backend, leaving the real adapter in place."""
        return patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=self.constructor)})

    def predict(self, source, **kwargs):
        assert source.dtype == np.uint8
        self.calls.append((source.copy(), kwargs))
        if tuple(source[0, 0]) == (0, 0, 255):
            rows = [([10, 10, 25, 25], 0.9, 1), ([40, 10, 55, 25], 0.8, 3)]
        else:
            rows = [([70, 10, 85, 25], 0.91, 1), ([40, 10, 55, 25], 0.8, 3)]
        return [SimpleNamespace(boxes=[Box(*row) for row in rows], orig_shape=source.shape[:2])]


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
        service = ComparisonService.for_real(settings)
        before = np.zeros((120, 160, 3), dtype=np.uint8)
        after = before.copy()
        before[0, 0] = (255, 0, 0)
        after[0, 0] = (0, 0, 255)
        b, a = self.root / "b.png", self.root / "a.png"
        Image.fromarray(before).save(b)
        Image.fromarray(after).save(a)
        with model.installed():
            result = service.compare(b, a, "real-test")
        model.constructor.assert_called_once()
        self.assertEqual(
            [(r["class_id"], r["status"]) for r in result["records"]],
            [(0, "moved"), (1, "unchanged")],
        )
        self.assertEqual(result["metadata"]["mode"], "real")
        self.assertEqual(result["metadata"]["checkpoint_sha256"], self.values["checkpoint_sha256"])
        self.assertEqual([x[0][0, 0].tolist() for x in model.calls], [[0, 0, 255], [255, 0, 0]])
        for _, options in model.calls:
            self.assertEqual(options["device"], "cpu")
            self.assertEqual(options["conf"], self.values["confidence_threshold"])
            self.assertEqual(options["iou"], self.values["nms_iou_threshold"])
            self.assertEqual(options["imgsz"], self.values["imgsz"])

    def test_checkpoint_changed_after_settings_validation_is_rejected_before_loading(self):
        service = ComparisonService.for_real(RealSettings.load(self.config))
        self.weight.write_bytes(b"changed after config verification")
        sample = self.root / "frame.png"
        Image.new("RGB", (160, 120)).save(sample)
        model = Model()
        with model.installed(), self.assertRaisesRegex(ValueError, "(?i)hash"):
            service.compare(sample, sample, "changed-weight")
        model.constructor.assert_not_called()
        self.assertEqual(model.calls, [])

    def test_model_loads_verified_snapshot_when_original_is_replaced_during_loading(self):
        settings = RealSettings.load(self.config)
        service = ComparisonService.for_real(settings)
        sample = self.root / "frame.png"
        Image.new("RGB", (160, 120)).save(sample)
        replacement = self.root / "replacement.pt"
        replacement.write_bytes(b"unverified replacement")
        model = Model()
        loaded_paths = []

        def load_checkpoint(path):
            # Deterministically simulate a path swap at the backend load boundary.
            replacement.replace(self.weight)
            snapshot = Path(path)
            self.assertEqual(
                hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                settings.checkpoint_sha256,
            )
            self.assertNotEqual(snapshot.resolve(), self.weight.resolve())
            loaded_paths.append(snapshot)
            return model

        model.constructor.side_effect = load_checkpoint
        with model.installed():
            first = service.compare(sample, sample, "snapshot")
            second = service.compare(sample, sample, "snapshot")
        model.constructor.assert_called_once()
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(first, second)
        self.assertEqual(first["metadata"]["checkpoint_sha256"], settings.checkpoint_sha256)
        self.assertEqual(self.weight.read_bytes(), b"unverified replacement")
        self.assertFalse(loaded_paths[0].exists())

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
        sample = self.root / "frame.png"
        Image.new("RGB", (20, 20)).save(sample)
        service = ComparisonService.for_real(RealSettings.load(self.config))
        with model.installed(), self.assertRaisesRegex(ValueError, "(?i)class|missing"):
            service.compare(sample, sample, "missing-class")
        self.assertEqual(model.calls, [])

    def test_fake_app_config_refuses_real_config(self):
        from deskcheck.application.app_config import AppConfig

        with self.assertRaises(ValueError):
            AppConfig.load(self.config)
