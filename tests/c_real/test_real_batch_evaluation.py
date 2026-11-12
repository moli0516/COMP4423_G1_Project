"""Part C real batch evaluation CLI must inject B's existing EvaluationRunner.

The only model used by tests is an explicitly injected API stub, NOT real YOLO.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from test_yolo_real_pipeline import Model

from deskcheck.detection.ultralytics_detector import UltralyticsDetector

ROOT = Path(__file__).resolve().parents[2]


class RealBatchEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.weights = self.folder / "best.pt"
        self.weights.write_bytes(b"STUB only - not a genuine trained checkpoint")
        self.checkpoint_sha = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.settings_file = self.folder / "real.json"
        self.settings_file.write_text(
            json.dumps({
                "schema_version": "1.0",
                "mode": "real",
                "checkpoint_path": "best.pt",
                "checkpoint_sha256": self.checkpoint_sha,
                "model_origin": "STUB-ONLY",
                "confidence_threshold": 0.25,
                "movement_threshold": 0.05,
                "nms_iou_threshold": 0.5,
                "imgsz": 640,
                "rules_version": "TEST-NOT-FROZEN",
                "class_names": ["bottle", "cup", "keyboard", "mouse", "book", "scissors"],
            }),
            encoding="utf-8",
        )
        before = np.zeros((120, 160, 3), dtype=np.uint8)
        after = before.copy()
        before[0, 0] = (255, 0, 0)
        after[0, 0] = (0, 0, 255)
        Image.fromarray(before).save(self.folder / "before.png")
        Image.fromarray(after).save(self.folder / "after.png")
        box_bottle_old = [10, 10, 25, 25]
        box_bottle_new = [70, 10, 85, 25]
        box_cup = [40, 10, 55, 25]
        self.truth = {
            "movement": [
                {"class_id": 0, "status": "moved",
                 "before_box": box_bottle_old, "after_box": box_bottle_new},
                {"class_id": 1, "status": "unchanged",
                 "before_box": box_cup, "after_box": box_cup},
            ],
            "control": [
                {"class_id": 0, "status": "unchanged",
                 "before_box": box_bottle_old, "after_box": box_bottle_old},
                {"class_id": 1, "status": "unchanged",
                 "before_box": box_cup, "after_box": box_cup},
            ],
        }
        self.manifest = {
            "schema_version": "1.0",
            "dataset_release": "STUB-not-a-real-dataset",
            "split": "val",
            "pairs": [
                {"pair_id": "movement", "before_path": "before.png",
                 "after_path": "after.png"},
                {"pair_id": "control", "before_path": "before.png",
                 "after_path": "before.png"},
            ],
            "ground_truth": self.truth,
            "eligible_no_change_pair_ids": ["control"],
        }
        self.manifest_file = self.folder / "manifest.json"
        self.update_manifest()
        self.output = self.folder / "output"

    def update_manifest(self):
        self.manifest_file.write_text(json.dumps(self.manifest), encoding="utf-8")

    def load_cli(self):
        path = ROOT / "scripts" / "evaluate_real_pairs.py"
        self.assertTrue(path.is_file(), "C real evaluation entry point is absent")
        spec = importlib.util.spec_from_file_location("c_real_eval", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def call_cli(self, module):
        argv = ["evaluate_real_pairs.py", "--config", str(self.settings_file),
                "--manifest", str(self.manifest_file), "--output", str(self.output)]
        with patch.object(sys, "argv", argv):
            return module.main()

    def test_batch_path_reuses_actual_B_evaluator_without_duplicating_scoring(self):
        cli = self.load_cli()
        model = Model()

        def detector_factory(settings):
            return UltralyticsDetector(settings, model=model)

        with patch.object(cli, "UltralyticsDetector", side_effect=detector_factory):
            self.assertEqual(self.call_cli(cli), 0)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"predictions.json", "metrics.json", "run_metadata.json"},
        )
        predictions = json.loads((self.output / "predictions.json").read_text())
        metadata = json.loads((self.output / "run_metadata.json").read_text())
        self.assertEqual([e["status"] for e in predictions["movement"]],
                         ["moved", "unchanged"])
        self.assertEqual([e["status"] for e in predictions["control"]],
                         ["unchanged", "unchanged"])
        self.assertEqual(metadata["pair_count"], 2)
        self.assertEqual(metadata["provenance"]["checkpoint_sha256"], self.checkpoint_sha)
        self.assertEqual(metadata["provenance"]["split"], "val")
        self.assertEqual(
            metadata["provenance"]["manifest_sha256"],
            hashlib.sha256(self.manifest_file.read_bytes()).hexdigest(),
        )
        self.assertEqual(metadata["provenance"]["mode"], "real")
        self.assertEqual(metadata["metric_scope"], "localized events and no-change alarms; excludes detector AP")
        self.assertFalse((self.output / "result.json").exists())

    def test_missing_checkpoint_fails_without_fake_fallback_or_outputs(self):
        cli = self.load_cli()
        self.weights.unlink()
        self.assertNotEqual(self.call_cli(cli), 0)
        self.assertFalse(self.output.exists())

    def test_incomplete_ground_truth_fails_before_model_loading(self):
        cli = self.load_cli()
        del self.manifest["ground_truth"]["control"]
        self.update_manifest()
        with patch.object(cli, "UltralyticsDetector",
                          side_effect=AssertionError("model must not be loaded")):
            self.assertNotEqual(self.call_cli(cli), 0)
        self.assertFalse(self.output.exists())

    def test_existing_target_is_never_overwritten(self):
        cli = self.load_cli()
        self.output.mkdir()
        proof = self.output / "sentinel.txt"
        proof.write_text("KEEP", encoding="utf-8")
        self.assertNotEqual(self.call_cli(cli), 0)
        self.assertEqual(proof.read_text(), "KEEP")
