"""Independent fixture oracle prevents plausible but incorrect synthetic demo exports."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deskcheck.application.demo_pair import make_demo_pair
from deskcheck.application.demo_verifier import verify_demo
from deskcheck.application.result_serializer import ResultSerializer

ROOT = Path(__file__).resolve().parents[2]


class DemoOracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.staging = Path(self.temporary.name) / "demo"
        cmd = [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "run_c_demo.py"),
            "--output",
            str(self.staging),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.before, self.after = make_demo_pair()

    def verify(self):
        return verify_demo(
            self.staging,
            self.before,
            self.after,
            ROOT / "configs" / "fake_app.json",
        )

    def test_genuine_demo_outputs_are_accepted(self):
        self.assertEqual(self.verify(), (4, 3))

    def test_plausible_translated_boxes_with_unchanged_statuses_are_rejected(self):
        result_path = self.staging / "bundle" / "result.json"
        csv_path = self.staging / "bundle" / "result.csv"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        moving = next(r for r in data["records"] if r["class_id"] == 0)
        # Move both boxes by the same amount. The status and distance remain valid,
        # but box locations no longer correspond to the fixture pixels.
        for side in ("before_box", "after_box"):
            moving[side][0] += 1.0
            moving[side][2] += 1.0
        # Keep transport internally consistent so this still exercises the
        # independent pixel/geometry oracle, rather than the serializer guard.
        for side in ("before", "after"):
            retained = next(r for r in data[f"{side}_detections"] if r["class_id"] == 0)
            retained["bbox_xyxy"] = moving[f"{side}_box"].copy()
        serializer = ResultSerializer()
        result_path.write_bytes(serializer.to_json(data).encode("utf-8"))
        csv_path.write_bytes(serializer.to_csv(data).encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "(?i)fixture|geometry|box"):
            self.verify()

    def test_readable_but_unannotated_png_is_rejected(self):
        # A valid 160x120 PNG must not masquerade as a rendered annotation.
        (self.staging / "bundle" / "annotated_before.png").write_bytes(self.before)
        with self.assertRaisesRegex(ValueError, "(?i)annotat|pixel"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
