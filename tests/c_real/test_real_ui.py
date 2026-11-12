"""UI real-mode gating: missing checkpoint cannot trigger fake inference."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from deskcheck.application.demo_pair import make_demo_pair

ROOT = Path(__file__).resolve().parents[2]


class RealUiTests(unittest.TestCase):
    def test_real_model_tab_exists_but_is_disabled_without_supplied_config(self):
        with patch.dict(os.environ, {"DESKCHECK_REAL_CONFIG": ""}):
            app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=25).run()
            app.radio[0].set_value("Run local YOLO checkpoint").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.button[0].disabled)
            self.assertEqual(len(app.get("download_button")), 0)

    def test_valid_checksum_but_unavailable_runtime_does_not_fallback_to_fake(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weight = root / "best.pt"
            weight.write_bytes(b"not a trained model; UI integration test only")
            config = root / "real.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "mode": "real",
                        "checkpoint_path": "best.pt",
                        "checkpoint_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
                        "model_origin": "test-stub",
                        "confidence_threshold": 0.25,
                        "movement_threshold": 0.05,
                        "nms_iou_threshold": 0.5,
                        "imgsz": 640,
                        "rules_version": "test-only",
                        "class_names": ["bottle", "cup", "keyboard", "mouse", "book", "scissors"],
                    }
                ),
                encoding="utf-8",
            )
            before, after = make_demo_pair()
            with patch.dict(os.environ, {"DESKCHECK_REAL_CONFIG": str(config)}):
                app = AppTest.from_file(
                    str(ROOT / "app/streamlit_app.py"), default_timeout=25
                ).run()
                app.radio[0].set_value("Run local YOLO checkpoint").run()
                app.file_uploader[0].set_value(("before.png", before, "image/png"))
                app.file_uploader[1].set_value(("after.png", after, "image/png")).run()
                self.assertFalse(app.button[0].disabled)
                with patch(
                    "deskcheck.detection.ultralytics_detector.UltralyticsDetector",
                    side_effect=RuntimeError("test model intentionally absent"),
                ):
                    app.button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertGreater(len(app.error), 0)
                self.assertEqual(len(app.json), 0)
                self.assertEqual(len(app.get("download_button")), 0)
