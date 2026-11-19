"""RealSettings source-binding regression: no stale provenance after direct construction or replace."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_yolo_real_pipeline import Model

from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.real_settings import RealSettings
from deskcheck.detection.ultralytics_detector import UltralyticsDetector


class RealSettingsProvenanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"fixture only -- no real trained model")
        self.content = {
            "schema_version": "1.0",
            "mode": "real",
            "checkpoint_path": "best.pt",
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "model_origin": "fixture-only",
            "confidence_threshold": 0.25,
            "movement_threshold": 0.05,
            "nms_iou_threshold": 0.5,
            "imgsz": 640,
            "rules_version": "test-v1",
            "class_names": ["bottle", "cup", "keyboard", "mouse", "book", "scissors"],
        }
        self.config_path = self.root / "real.json"
        self.config_path.write_text(json.dumps(self.content), encoding="utf-8")

    def test_load_is_immutable_and_repeatable_with_same_provenance(self):
        settings = RealSettings.load(self.config_path)
        self.assertEqual(
            settings.config_sha256, hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        )
        self.assertEqual(replace(settings), settings)
        self.assertEqual(settings.checkpoint_path, self.checkpoint.resolve())

    def test_replacing_effective_parameters_must_not_keep_stale_source_hash(self):
        settings = RealSettings.load(self.config_path)
        changes = {
            "confidence_threshold": 0.7,
            "movement_threshold": 0.3,
            "nms_iou_threshold": 0.9,
            "imgsz": 320,
            "rules_version": "test-v2",
            "model_origin": "different-source",
            "mode": "fake",
            "config_sha256": "0" * 64,
            "checkpoint_sha256": "0" * 64,
            "checkpoint_path": self.root / "other.pt",
        }
        for name, value in changes.items():
            with self.subTest(field=name), self.assertRaises((TypeError, ValueError)):
                replace(settings, **{name: value})

    def test_direct_constructor_with_forged_effective_settings_is_rejected(self):
        settings = RealSettings.load(self.config_path)
        payload = {
            "checkpoint_path": settings.checkpoint_path,
            "checkpoint_sha256": settings.checkpoint_sha256,
            "config_sha256": settings.config_sha256,
            "model_origin": settings.model_origin,
            "confidence_threshold": 0.88,  # not in hashed source
            "movement_threshold": settings.movement_threshold,
            "nms_iou_threshold": settings.nms_iou_threshold,
            "imgsz": settings.imgsz,
            "rules_version": settings.rules_version,
            "mode": settings.mode,
        }
        with self.assertRaises((TypeError, ValueError)):
            RealSettings(**payload)

    def test_real_pipeline_cannot_be_built_from_replaced_settings(self):
        settings = RealSettings.load(self.config_path)
        with self.assertRaises((TypeError, ValueError)):
            bad = replace(settings, movement_threshold=0.7)
            ComparisonService(config=bad, detector=UltralyticsDetector(bad, model=Model()))

    def test_direct_constructor_rejects_invalid_rehashed_source_values(self):
        settings = RealSettings.load(self.config_path)
        for name, value in (
            ("confidence_threshold", 1.5),
            ("movement_threshold", -0.01),
            ("imgsz", 0),
            ("rules_version", ""),
            ("model_origin", ""),
        ):
            with self.subTest(field=name):
                changed_source = {**self.content, name: value}
                payload = json.dumps(changed_source).encode("utf-8")
                with self.assertRaises(ValueError):
                    replace(
                        settings,
                        **{
                            name: value,
                            "_source_bytes": payload,
                            "config_sha256": hashlib.sha256(payload).hexdigest(),
                        },
                    )

    def test_source_changed_on_disk_is_not_silently_mislabeled_as_old_digest(self):
        settings = RealSettings.load(self.config_path)
        self.content["movement_threshold"] = 0.25
        self.config_path.write_text(json.dumps(self.content), encoding="utf-8")
        reloaded = RealSettings.load(self.config_path)
        self.assertNotEqual(settings.config_sha256, reloaded.config_sha256)
        self.assertEqual(settings.movement_threshold, 0.05)
        self.assertEqual(reloaded.movement_threshold, 0.25)
