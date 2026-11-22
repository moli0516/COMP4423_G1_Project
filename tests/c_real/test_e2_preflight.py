"""Guard against starting the ML backend during the E2 configuration check."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = runpy.run_path(str(ROOT / "scripts/preflight_e2.py"))


@pytest.mark.parametrize("has_backend", [False, True])
def test_real_cli_probe_preserves_backend_settings(tmp_path, monkeypatch, has_backend):
    original_settings = {"runs_dir": "my-existing-runs"}
    backend = SimpleNamespace(settings=original_settings) if has_backend else None
    monkeypatch.setitem(sys.modules, "ultralytics", backend)
    for folder in ("images/train", "images/val", "labels/train", "labels/val"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "dataset.yaml").write_text(f"path: {tmp_path.as_posix()}\n")
    config = json.loads((ROOT / "configs/experiments/baseline.json").read_bytes())
    config["training"]["device"] = "cpu"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(config))
    cli = runpy.run_path(str(ROOT / "scripts/train.py"))["main"]
    weights, kwargs = PREFLIGHT["probe"](cli, candidate, tmp_path)
    assert original_settings == {"runs_dir": "my-existing-runs"}
    assert weights == "yolov8n.pt"
    assert (kwargs["imgsz"], kwargs["batch"]) == (640, 16)
    assert list((tmp_path / "candidate").iterdir()) == []


def test_backend_default_drift_is_detected():
    kwargs = {"imgsz": 960, "batch": 16, "device": 0, "optimizer": "SGD"}
    recorded = {
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "model": "yolov8n.pt",
        "optimizer": "SGD",
        "deterministic": True,
    }
    with pytest.raises(ValueError, match="deterministic"):
        PREFLIGHT["check_effective_args"](
            "yolov8n.pt", kwargs, recorded, {"deterministic": False}, 960, 16
        )
