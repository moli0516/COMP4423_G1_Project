"""Part C boundary, presentation and adversarial-input acceptance."""

import copy
import importlib
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.demo_pair import make_demo_pair
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.detection.fake_color_detector import FakeColorDetector
from deskcheck.domain.image_frame import ImageFrame

ROOT = Path(__file__).resolve().parents[2]


class ReleaseCoreTests(unittest.TestCase):
    def module(self, name):
        full = "deskcheck.application." + name
        self.assertIsNotNone(importlib.util.find_spec(full), full + " is not implemented")
        return importlib.import_module(full)

    def pipeline(self, detector=None):
        return ComparisonService(
            detector=detector or FakeColorDetector(),
            config=AppConfig.load(ROOT / "configs/fake_app.json"),
        )

    def test_detector_receives_shared_frame_and_b_results(self):
        seen = []

        class Detector:
            def predict(self, frame):
                seen.append(frame)
                return FakeColorDetector().predict(frame)

        before, after = make_demo_pair()
        frames = ImagePairLoader().load(before, after, "shared")
        actual = self.pipeline(Detector()).compare_frames(*frames, pair_id="shared")
        self.assertEqual(actual, self.pipeline().compare(before, after, "shared"))
        self.assertEqual(len(seen), 2)
        for frame, delivered in zip(frames, seen):
            self.assertIs(type(delivered), ImageFrame)
            self.assertIs(delivered, frame)

    def test_display_filter_preserves_complete_export_and_correspondence(self):
        mod = self.module("presentation")
        before, after = make_demo_pair()
        result = self.pipeline().compare(before, after, "view")
        saved = copy.deepcopy(result)
        rows = mod.comparison_rows(result)
        filtered = mod.comparison_rows(result, changes_only=True)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(filtered), 3)
        self.assertEqual({r["Status"] for r in rows}, {"added", "removed", "moved", "unchanged"})
        self.assertEqual({r["Object"] for r in rows}, {"bottle", "cup", "keyboard", "mouse"})
        self.assertEqual(len({r["Object ID"] for r in rows}), 4)
        self.assertEqual(result, saved)
        self.assertTrue(all(r["Status"] != "unchanged" for r in filtered))
        self.assertEqual(
            mod.comparison_rows(
                {**result, "records": [], "before_detections": [], "after_detections": []}
            ),
            [],
        )

    def test_wipe_endpoints_dimensions_and_no_mutation(self):
        wipe = self.module("presentation").comparison_wipe
        before = Image.new("RGB", (20, 10), (10, 20, 30))
        after = Image.new("RGB", (20, 10), (100, 110, 120))
        saved = before.tobytes(), after.tobytes()
        self.assertEqual(wipe(before, after, 0).tobytes(), after.tobytes())
        self.assertEqual(wipe(before, after, 100).tobytes(), before.tobytes())
        middle = wipe(before, after, 50)
        self.assertEqual(middle.getpixel((2, 4)), before.getpixel((2, 4)))
        self.assertEqual(middle.getpixel((17, 4)), after.getpixel((17, 4)))
        self.assertEqual((before.tobytes(), after.tobytes()), saved)
        for percent in (-1, 101, float("nan"), True):
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                wipe(before, after, percent)
        with self.assertRaises(ValueError):
            wipe(before, Image.new("RGB", (21, 10)), 50)

    def test_six_classes_reverse_pair_and_source_identity_independence(self):
        import io

        pixels = np.full((80, 180, 3), 240, dtype=np.uint8)
        for cid, color in enumerate(FakeColorDetector.COLORS):
            pixels[5:20, 5 + 25 * cid : 20 + 25 * cid] = color
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        identical = self.pipeline().compare(buf.getvalue(), buf.getvalue(), "six")
        self.assertEqual(len(identical["records"]), 6)
        self.assertTrue(all(r["status"] == "unchanged" for r in identical["records"]))
        a, b = make_demo_pair()
        forward = self.pipeline().compare(a, b, "before")
        reverse = self.pipeline().compare(b, a, "after")
        inverted = {
            "added": "removed",
            "removed": "added",
            "moved": "moved",
            "unchanged": "unchanged",
        }
        by_id = {r["class_id"]: r for r in reverse["records"]}
        for r in forward["records"]:
            other = by_id[r["class_id"]]
            self.assertEqual(other["status"], inverted[r["status"]])
            self.assertEqual(other["before_box"], r["after_box"])
            self.assertEqual(other["after_box"], r["before_box"])


if __name__ == "__main__":
    unittest.main()
