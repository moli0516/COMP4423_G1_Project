"""End-user C interface acceptance against the real Streamlit app."""

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


class ReleaseUiTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_table_summary_filter_and_full_download_payload_survive_rerun(self):
        app = self.app()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe), 1, "human-readable result table is missing")
        self.assertEqual(len(app.dataframe[0].value), 4)
        self.assertEqual([m.value for m in app.metric], ["4", "3"])
        self.assertGreaterEqual(len(app.checkbox), 1)
        original = app.json[0].value
        app.checkbox[0].check().run()
        self.assertEqual(len(app.dataframe[0].value), 3)
        self.assertEqual(json.loads(app.json[0].value), json.loads(original))
        self.assertEqual(len(app.get("download_button")), 4)

    def test_pair_id_and_source_change_remove_old_results(self):
        app = self.app()
        app.button[0].click().run()
        self.assertEqual(len(app.json), 1)
        app.text_input[0].set_value("changed-id").run()
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        app.button[0].click().run()
        self.assertEqual(json.loads(app.json[0].value)["pair_id"], "changed-id")
        app.radio[0].set_value("Inspect real photos (no AI)").run()
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("no" in x.value.lower() for x in app.warning))

    def test_original_and_wipe_views_are_available(self):
        app = self.app()
        app.button[0].click().run()
        self.assertGreaterEqual(len(app.radio), 2, "result views missing")
        app.radio[1].set_value("Originals").run()
        self.assertEqual(len(app.exception), 0)
        app.radio[1].set_value("Before / after wipe").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.slider), 1)
        app.slider[0].set_value(0).run()
        app.slider[0].set_value(100).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()


class UploadUiTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
        app.radio[0].set_value("Upload synthetic images").run()
        return app

    def test_upload_corruption_recovery_and_stale_download_removal(self):
        from deskcheck.application.demo_pair import make_demo_pair

        a, b = make_demo_pair()
        app = self.make_app()
        app.file_uploader[0].set_value(("before.png", a, "image/png"))
        app.file_uploader[1].set_value(("after.png", b, "image/png")).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.json), 1)
        self.assertEqual(len(app.download_button), 4)
        app.file_uploader[1].set_value(("after.png", b"broken", "image/png")).run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.error), 0)
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.download_button), 0)
        self.assertTrue(app.button[0].disabled)
        app.file_uploader[1].set_value(("after.png", b, "image/png")).run()
        self.assertFalse(app.button[0].disabled)
        app.button[0].click().run()
        self.assertEqual(len(app.json), 1)
        app.file_uploader[0].clear().run()
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.download_button), 0)

    def test_alpha_dimension_and_real_photo_inspection_are_explicit(self):
        import io

        from PIL import Image

        from deskcheck.application.demo_pair import make_demo_pair

        a, b = make_demo_pair()
        app = self.make_app()
        app.file_uploader[0].set_value(("before.png", a, "image/png"))
        for image in (Image.new("RGBA", (160, 120)), Image.new("RGB", (9, 8))):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            app.file_uploader[1].set_value(("bad.png", buffer.getvalue(), "image/png")).run()
            self.assertEqual(len(app.exception), 0)
            self.assertGreater(len(app.error), 0)
            self.assertTrue(app.button[0].disabled)
        app.radio[0].set_value("Inspect real photos (no AI)").run()
        app.file_uploader[0].set_value(("photo1.png", a, "image/png"))
        app.file_uploader[1].set_value(("photo2.png", b, "image/png")).run()
        self.assertEqual(len(app.error), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.download_button), 0)
