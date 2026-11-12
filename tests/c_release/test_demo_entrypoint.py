"""Smoke entry point must work without manually sourcing fixture images."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class DemoEntrypointTests(unittest.TestCase):
    def test_demo_publishes_real_cli_bundle_and_refuses_existing_target(self):
        script = ROOT / "scripts/run_c_demo.py"
        self.assertTrue(script.exists(), "one-command demo entry point is missing")
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "demo"
            cmd = [sys.executable, "-B", str(script), "--output", str(target)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                {x.name for x in target.iterdir()}, {"before.png", "after.png", "bundle"}
            )
            self.assertEqual(
                {x.name for x in (target / "bundle").iterdir()},
                {"result.json", "result.csv", "annotated_before.png", "annotated_after.png"},
            )
            data = json.loads((target / "bundle/result.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["records"]), 4)
            self.assertEqual(sum(r["indicates_change"] for r in data["records"]), 3)
            from deskcheck.application.result_serializer import ResultSerializer

            serializer = ResultSerializer()
            self.assertEqual(
                (target / "bundle/result.json").read_bytes(),
                serializer.to_json(data).encode("utf-8"),
            )
            self.assertEqual(
                (target / "bundle/result.csv").read_bytes(), serializer.to_csv(data).encode("utf-8")
            )
            saved = (target / "bundle/result.json").read_bytes()
            again = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(again.returncode, 2)
            self.assertEqual((target / "bundle/result.json").read_bytes(), saved)
