"""Same synthetic pair must have identical results in UI, CLI and B runner."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.demo_pair import make_demo_pair
from deskcheck.detection.fake_color_detector import FakeColorDetector
from deskcheck.evaluation.evaluation_runner import EvaluationRunner

ROOT = Path(__file__).resolve().parents[2]


class ThreeEntrypointParityTests(unittest.TestCase):
    def test_one_pair_agrees_across_ui_cli_service_and_b_evaluation(self):
        before_bytes, after_bytes = make_demo_pair()
        config_path = ROOT / "configs" / "fake_app.json"
        service = ComparisonService(
            detector=FakeColorDetector(),
            config=AppConfig.load(config_path),
        )
        direct = service.compare(before_bytes, after_bytes, "c-demo")
        self.assertEqual([r["status"] for r in direct["records"]],
                         ["moved", "removed", "unchanged", "added"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before_path = root / "before.png"
            after_path = root / "after.png"
            before_path.write_bytes(before_bytes)
            after_path.write_bytes(after_bytes)
            cmd = [
                sys.executable, "-B", str(ROOT / "scripts/predict_pair.py"),
                "--config", str(config_path),
                "--before", str(before_path),
                "--after", str(after_path),
                "--pair-id", "c-demo",
                "--output", str(root / "cli"),
            ]
            run = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            cli = json.loads((root / "cli/result.json").read_text(encoding="utf-8"))
            self.assertEqual(cli, direct)

            app = AppTest.from_file(
                str(ROOT / "app/streamlit_app.py"), default_timeout=30
            ).run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.json), 1)
            self.assertEqual(json.loads(app.json[0].value), direct)

            # Truth is independently specified, never sent to ComparisonService.
            manual_demo_truth = [
                {"class_id": 0, "status": "moved",
                 "before_box": [10, 10, 25, 25], "after_box": [80, 10, 95, 25]},
                {"class_id": 1, "status": "removed",
                 "before_box": [10, 40, 25, 55], "after_box": None},
                {"class_id": 2, "status": "unchanged",
                 "before_box": [10, 70, 25, 85], "after_box": [10, 70, 25, 85]},
                {"class_id": 3, "status": "added",
                 "before_box": None, "after_box": [80, 40, 95, 55]},
            ]
            manual_control_truth = [
                {"class_id": 0, "status": "unchanged",
                 "before_box": [10, 10, 25, 25], "after_box": [10, 10, 25, 25]},
                {"class_id": 1, "status": "unchanged",
                 "before_box": [10, 40, 25, 55], "after_box": [10, 40, 25, 55]},
                {"class_id": 2, "status": "unchanged",
                 "before_box": [10, 70, 25, 85], "after_box": [10, 70, 25, 85]},
            ]
            evaluation = EvaluationRunner(service).run(
                [
                    {"pair_id": "c-demo", "before_path": before_path, "after_path": after_path},
                    {"pair_id": "control", "before_path": before_path, "after_path": before_path},
                ],
                {"c-demo": manual_demo_truth, "control": manual_control_truth},
                ["control"],
                root / "evaluation",
                metadata={"mode": "fake", "dataset": "synthetic-fixture-only"},
            )
            self.assertEqual(evaluation["predictions"]["c-demo"], direct["records"])
            self.assertEqual(evaluation["predictions"]["control"],
                             service.compare(before_bytes, before_bytes, "control")["records"])
            self.assertEqual(set(evaluation["predictions"]), {"c-demo", "control"})
            self.assertEqual(
                json.loads((root / "evaluation/predictions.json").read_text(encoding="utf-8")),
                evaluation["predictions"],
            )


if __name__ == "__main__":
    unittest.main()
