"""Real CLI must refuse a missing checkpoint and never silently use synthetic mode."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from test_yolo_real_pipeline import Model, RealIntegration

from deskcheck.detection.ultralytics_detector import UltralyticsDetector

ROOT = Path(__file__).resolve().parents[2]


class RealCliTests(unittest.TestCase):
    def load_cli(self):
        path = ROOT / "scripts" / "predict_real_pair.py"
        self.assertTrue(path.exists(), "real CLI implementation is missing")
        spec = importlib.util.spec_from_file_location("real_predict_cli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_missing_checkpoint_never_publishes_synthetic_result(self):
        cli = self.load_cli()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "real.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "mode": "real",
                        "checkpoint_path": "not-a-checkpoint.pt",
                        "checkpoint_sha256": "a" * 64,
                        "model_origin": "unknown",
                        "confidence_threshold": 0.25,
                        "movement_threshold": 0.05,
                        "nms_iou_threshold": 0.5,
                        "imgsz": 640,
                        "rules_version": "unapproved",
                        "class_names": ["bottle", "cup", "keyboard", "mouse", "book", "scissors"],
                    }
                ),
                encoding="utf-8",
            )
            before = root / "before.png"
            after = root / "after.png"
            Image.new("RGB", (16, 16)).save(before)
            Image.new("RGB", (16, 16)).save(after)
            output = root / "output"
            argv = [
                "predict_real_pair.py",
                "--config",
                str(config),
                "--before",
                str(before),
                "--after",
                str(after),
                "--pair-id",
                "r1",
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                code = cli.main()
            self.assertNotEqual(code, 0)
            self.assertFalse(output.exists())

    def test_real_cli_exports_b_records_with_explicit_injected_model(self):
        cli = self.load_cli()
        helper = RealIntegration()
        helper.setUp()
        try:
            before = np.zeros((120, 160, 3), dtype=np.uint8)
            after = before.copy()
            before[0, 0] = (255, 0, 0)
            after[0, 0] = (0, 0, 255)
            b, a = helper.root / "before.png", helper.root / "after.png"
            Image.fromarray(before).save(b)
            Image.fromarray(after).save(a)
            output = helper.root / "real-bundle"
            argv = [
                "predict_real_pair.py",
                "--config",
                str(helper.config),
                "--before",
                str(b),
                "--after",
                str(a),
                "--pair-id",
                "pair1",
                "--output",
                str(output),
            ]

            def stub_detector(settings):
                return UltralyticsDetector(settings, model=Model())

            with (
                patch.object(cli, "UltralyticsDetector", side_effect=stub_detector),
                patch.object(sys, "argv", argv),
            ):
                self.assertEqual(cli.main(), 0)
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {
                    "result.json",
                    "result.csv",
                    "annotated_before.png",
                    "annotated_after.png",
                },
            )
            data = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(data["metadata"]["mode"], "real")
            self.assertEqual(data["pair_id"], "pair1")
            self.assertEqual([r["status"] for r in data["records"]], ["moved", "unchanged"])
        finally:
            helper.doCleanups()
