"""Behavior tests for the bounded C synthetic integration."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class CApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, array):
        path = self.root / name
        Image.fromarray(array).save(path)
        return path

    def setup_pipeline(self):
        from deskcheck.application.app_config import AppConfig
        from deskcheck.application.comparison_service import ComparisonService
        from deskcheck.detection.fake_color_detector import FakeColorDetector

        config = AppConfig.load(Path(__file__).resolve().parents[2] / "configs" / "fake_app.json")
        return ComparisonService(detector=FakeColorDetector(), config=config)

    def mixed(self):
        from deskcheck.detection.fake_color_detector import FakeColorDetector

        before = np.full((120, 160, 3), 240, dtype=np.uint8)
        after = before.copy()
        colors = FakeColorDetector.COLORS
        before[10:25, 10:25] = colors[0]
        after[10:25, 80:95] = colors[0]
        before[40:55, 10:25] = colors[1]
        before[70:85, 10:25] = colors[2]
        after[70:85, 10:25] = colors[2]
        after[40:55, 80:95] = colors[3]
        return self.image("before.png", before), self.image("after.png", after)

    def test_real_b_chain_produces_four_statuses(self):
        service = self.setup_pipeline()
        before, after = self.mixed()
        hashes = (before.read_bytes(), after.read_bytes())
        result = service.compare(before, after, "mixed")
        self.assertEqual(result["pair_id"], "mixed")
        self.assertEqual(result["image_size"], [160, 120])
        self.assertEqual(
            {r["class_id"]: r["status"] for r in result["records"]},
            {0: "moved", 1: "removed", 2: "unchanged", 3: "added"},
        )
        self.assertIsNone(result["records"][1]["after_box"])
        self.assertIsNone(result["records"][3]["before_box"])
        self.assertEqual((before.read_bytes(), after.read_bytes()), hashes)
        self.assertEqual(result["metadata"]["mode"], "fake")
        self.assertIsNone(result["metadata"]["checkpoint_sha256"])

    def test_empty_and_one_sided_scenes(self):
        service = self.setup_pipeline()
        blank = np.full((120, 160, 3), 240, dtype=np.uint8)
        empty = self.image("empty.png", blank)
        before, after = self.mixed()
        self.assertEqual(service.compare(empty, empty, "empty")["records"], [])
        self.assertTrue(
            all(r["status"] == "added" for r in service.compare(empty, after, "added")["records"])
        )
        self.assertTrue(
            all(
                r["status"] == "removed"
                for r in service.compare(before, empty, "removed")["records"]
            )
        )

    def test_loader_applies_exif_without_resize(self):
        from deskcheck.application.image_pair_loader import ImagePairLoader

        arr = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        image = Image.fromarray(arr)
        exif = image.getexif()
        exif[274] = 6
        source = self.root / "oriented.png"
        image.save(source, exif=exif)
        expected = np.rot90(arr, k=3)
        other = self.image("canonical.png", expected)
        first, second = ImagePairLoader().load(source, other, "orientation")
        np.testing.assert_array_equal(first.array, expected)
        np.testing.assert_array_equal(second.array, expected)
        self.assertEqual((first.width, first.height), (3, 4))
        self.assertFalse(first.array.flags.writeable)

    def test_loader_converts_greyscale_and_rejects_alpha(self):
        from deskcheck.application.image_pair_loader import ImagePairLoader

        gray = self.root / "gray.png"
        Image.new("L", (8, 7), 90).save(gray)
        a, b = ImagePairLoader().load(gray, gray, "gray")
        self.assertEqual(a.array.shape, (7, 8, 3))
        self.assertEqual(a.array.dtype, np.uint8)
        np.testing.assert_array_equal(a.array, b.array)
        alpha = self.root / "alpha.png"
        Image.new("RGBA", (8, 7), (1, 2, 3, 0)).save(alpha)
        with self.assertRaisesRegex(ValueError, "(?i)transparen"):
            ImagePairLoader().load(alpha, gray, "alpha")

    def test_loader_rejects_missing_corrupt_and_dimension_mismatch(self):
        from deskcheck.application.image_pair_loader import ImagePairLoader

        normal = self.image("normal.png", np.zeros((10, 10, 3), dtype=np.uint8))
        other = self.image("other.png", np.zeros((11, 10, 3), dtype=np.uint8))
        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        for path in (self.root / "missing.png", corrupt, other):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ImagePairLoader().load(normal, path, "bad")

    def test_service_propagates_detector_failure(self):
        from deskcheck.application.app_config import AppConfig
        from deskcheck.application.comparison_service import ComparisonService

        class BrokenDetector:
            def predict(self, image):
                raise RuntimeError("detector failed")

        config = AppConfig.load(Path(__file__).resolve().parents[2] / "configs" / "fake_app.json")
        service = ComparisonService(detector=BrokenDetector(), config=config)
        blank = self.image("blank.png", np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, "detector failed"):
            service.compare(blank, blank, "bad")

    def test_config_rejects_unknown_duplicate_and_real_modes(self):
        from deskcheck.application.app_config import AppConfig

        config_path = Path(__file__).resolve().parents[2] / "configs" / "fake_app.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        for update in (
            {"unexpected": 1},
            {"mode": "real"},
            {"confidence_threshold": True},
            {"movement_threshold": None},
            {"class_names": ["wrong"]},
        ):
            path = self.root / "bad.json"
            path.write_text(json.dumps({**raw, **update}), encoding="utf-8")
            with self.subTest(update=update), self.assertRaises(ValueError):
                AppConfig.load(path)
        path.write_text('{"mode":"fake","mode":"real"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            AppConfig.load(path)

    def test_json_csv_and_renderer_preserve_originals(self):
        from deskcheck.application.image_pair_loader import ImagePairLoader
        from deskcheck.application.result_renderer import ResultRenderer
        from deskcheck.application.result_serializer import ResultSerializer

        service = self.setup_pipeline()
        before, after = self.mixed()
        result = service.compare(before, after, "mixed")
        first, second = ImagePairLoader().load(before, after, "mixed")
        originals = (first.array.copy(), second.array.copy())
        images = ResultRenderer().render(first, second, result)
        self.assertEqual([im.size for im in images], [(160, 120), (160, 120)])
        self.assertFalse(np.array_equal(np.asarray(images[0]), originals[0]))
        np.testing.assert_array_equal(first.array, originals[0])
        np.testing.assert_array_equal(second.array, originals[1])
        encoded = ResultSerializer().to_json(result)
        self.assertEqual(json.loads(encoded), result)
        rows = list(csv.DictReader(io.StringIO(ResultSerializer().to_csv(result))))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["after_x1"], "")
        result["metadata"]["bad"] = float("nan")
        with self.assertRaises(ValueError):
            ResultSerializer().to_json(result)

    def test_runner_accepts_real_c_service_and_fresh_output(self):
        from deskcheck.evaluation.evaluation_runner import EvaluationRunner

        service = self.setup_pipeline()
        before, after = self.mixed()
        # Independent manual event expectations, never supplied to the detector/service.
        truth = {
            "mixed": [
                {
                    "class_id": 0,
                    "status": "moved",
                    "before_box": [10, 10, 25, 25],
                    "after_box": [80, 10, 95, 25],
                },
                {
                    "class_id": 1,
                    "status": "removed",
                    "before_box": [10, 40, 25, 55],
                    "after_box": None,
                },
                {
                    "class_id": 2,
                    "status": "unchanged",
                    "before_box": [10, 70, 25, 85],
                    "after_box": [10, 70, 25, 85],
                },
                {
                    "class_id": 3,
                    "status": "added",
                    "before_box": None,
                    "after_box": [80, 40, 95, 55],
                },
            ],
            "control": [
                {
                    "class_id": 2,
                    "status": "unchanged",
                    "before_box": [10, 70, 25, 85],
                    "after_box": [10, 70, 25, 85],
                }
            ],
        }
        control = np.full((120, 160, 3), 240, dtype=np.uint8)
        from deskcheck.detection.fake_color_detector import FakeColorDetector

        control[70:85, 10:25] = FakeColorDetector.COLORS[2]
        c1 = self.image("control_before.png", control)
        c2 = self.image("control_after.png", control)
        output = self.root / "evaluation"
        result = EvaluationRunner(service).run(
            [
                {"pair_id": "mixed", "before_path": before, "after_path": after},
                {"pair_id": "control", "before_path": c1, "after_path": c2},
            ],
            truth,
            ["control"],
            output,
            metadata={"mode": "fake", "code_base": "146c8cd"},
        )
        self.assertEqual(set(result["predictions"]), {"control", "mixed"})
        self.assertEqual(
            {p.name for p in output.iterdir()},
            {"predictions.json", "metrics.json", "run_metadata.json"},
        )
        with self.assertRaises(FileExistsError):
            EvaluationRunner(service).run(
                [
                    {"pair_id": "mixed", "before_path": before, "after_path": after},
                    {"pair_id": "control", "before_path": c1, "after_path": c2},
                ],
                truth,
                ["control"],
                output,
            )

    def test_frame_owns_pixels_and_has_no_source_identity(self):
        from deskcheck.application.image_frame import ImageFrame

        pixels = np.zeros((8, 9, 3), dtype=np.uint8)
        frame = ImageFrame(pixels)
        pixels[:] = 255
        self.assertFalse(frame.array.any())
        self.assertFalse(hasattr(frame, "source_path"))
        with self.assertRaises(ValueError):
            frame.array[0, 0] = [1, 2, 3]

    def test_service_accepts_uploaded_bytes_and_rejects_bad_pair_id(self):
        service = self.setup_pipeline()
        before, after = self.mixed()
        self.assertEqual(
            service.compare(before.read_bytes(), after.read_bytes(), "bytes")["pair_id"], "bytes"
        )
        with self.assertRaises(ValueError):
            service.compare(before, after, " ")

    def test_cli_publishes_complete_bundle_and_refuses_overwrite(self):
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[2]
        before, after = self.mixed()
        output = self.root / "bundle"
        command = [
            sys.executable,
            "-B",
            str(root / "scripts" / "predict_pair.py"),
            "--config",
            str(root / "configs" / "fake_app.json"),
            "--before",
            str(before),
            "--after",
            str(after),
            "--pair-id",
            "cli",
            "--output",
            str(output),
        ]
        run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(
            {p.name for p in output.iterdir()},
            {"result.json", "result.csv", "annotated_before.png", "annotated_after.png"},
        )
        self.assertEqual(len(json.loads((output / "result.json").read_text())["records"]), 4)
        saved = {p.name: p.read_bytes() for p in output.iterdir()}
        again = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(again.returncode, 2, again.stdout + again.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, saved)
        with Image.open(output / "annotated_before.png") as image:
            self.assertEqual(image.size, (160, 120))

    def test_frame_cannot_restore_writeability(self):
        from deskcheck.application.image_frame import ImageFrame

        frame = ImageFrame(np.zeros((8, 9, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            frame.array.setflags(write=True)

    def test_configuration_changes_cannot_keep_old_provenance(self):
        from dataclasses import replace

        service = self.setup_pipeline()
        with self.assertRaises(ValueError):
            replace(service.config, movement_threshold=1.0)
        with self.assertRaises(AttributeError):
            service.config = service.config

    def test_cli_uses_exactly_one_decoded_pixel_snapshot(self):
        import importlib.util
        from unittest.mock import patch

        from deskcheck.application.image_pair_loader import ImagePairLoader

        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "c_cli_snapshot", root / "scripts" / "predict_pair.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        before, after = self.mixed()
        frames = ImagePairLoader().load(before, after, "snapshot")
        blank = self.image("blank_snapshot.png", np.full((120, 160, 3), 240, dtype=np.uint8))
        other_frames = ImagePairLoader().load(blank, blank, "snapshot")
        argv = [
            "predict_pair.py",
            "--config",
            str(root / "configs" / "fake_app.json"),
            "--before",
            str(before),
            "--after",
            str(after),
            "--pair-id",
            "snapshot",
            "--output",
            str(self.root / "snapshot-output"),
        ]
        with patch.object(ImagePairLoader, "load", side_effect=[frames, other_frames]) as load:
            with patch("sys.argv", argv):
                self.assertEqual(cli.main(), 0)
            self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
