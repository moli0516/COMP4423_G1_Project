"""Headless checks with an existing venv, never install dependencies."""

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Streamlit not installed")
class CUiTests(unittest.TestCase):
    def test_synthetic_demo_runs_and_input_change_clears_result(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[2]
        app = AppTest.from_file(str(root / "app" / "streamlit_app.py"), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.json), 1)
        for status in ("moved", "removed", "added", "unchanged"):
            self.assertIn(status, app.json[0].value)
        app.radio[0].set_value("Upload synthetic images").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.json), 0)


if __name__ == "__main__":
    unittest.main()
