"""Fail-closed acceptance tests for the one-command synthetic demo launcher."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_c_demo.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("jerry_c_demo_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DemoTruthfulnessTests(unittest.TestCase):
    def test_success_exit_without_output_must_not_publish_or_claim_success(self):
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "demo"
            with (
                patch.object(sys, "argv", [str(SCRIPT), "--output", str(target)]),
                patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ok", "")),
            ):
                result = launcher.main()
            self.assertEqual(result, 1, "A successful child exit without deliverables is not a complete demo")
            self.assertFalse(target.exists(), "Incomplete results must not be published")

    def test_success_exit_with_wrong_counts_must_not_publish(self):
        launcher = load_launcher()

        def lying_child(command, **kwargs):
            output_dir = Path(command[command.index("--output") + 1])
            output_dir.mkdir()
            (output_dir / "result.json").write_text(
                json.dumps({
                    "pair_id": "c-demo",
                    "image_size": [160, 120],
                    "records": [],
                    "metadata": {"mode": "fake", "checkpoint_sha256": None},
                }),
                encoding="utf-8",
            )
            (output_dir / "result.csv").write_text("pair_id\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "mode=fake records=4 changes=3", "")

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "demo"
            with (
                patch.object(sys, "argv", [str(SCRIPT), "--output", str(target)]),
                patch.object(launcher.subprocess, "run", side_effect=lying_child),
            ):
                result = launcher.main()
            self.assertEqual(result, 1, "The launcher must inspect actual JSON, not rely on stdout")
            self.assertFalse(target.exists(), "Wrong content must not be published")


if __name__ == "__main__":
    unittest.main()
