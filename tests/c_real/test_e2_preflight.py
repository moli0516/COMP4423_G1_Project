"""Guard against starting the ML backend during the E2 configuration check."""

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = runpy.run_path(str(ROOT / "scripts/preflight_e2.py"))


def test_real_cli_reaches_spy_without_backend_or_weights(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    for folder in ("images/train", "images/val", "labels/train", "labels/val"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "dataset.yaml").write_text(f"path: {tmp_path.as_posix()}\n")
    config = json.loads((ROOT / "configs/experiments/baseline.json").read_bytes())
    config["training"]["device"] = "cpu"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(config))
    cli = runpy.run_path(str(ROOT / "scripts/train.py"))["main"]
    weights, kwargs = PREFLIGHT["probe"](cli, candidate, tmp_path)
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
