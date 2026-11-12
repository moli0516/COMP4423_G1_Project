"""Read-only startup diagnostics, including corrupt/missing local artifacts."""

import hashlib
import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("c_setup", ROOT / "scripts/check_c_setup.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


@pytest.fixture(autouse=True)
def pip_success(monkeypatch):
    # The CLI smoke executes actual pip check. Unit cases isolate each diagnostic.
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="No broken requirements found.", stderr=""
    ))


def test_current_synthetic_setup_is_ready_without_claiming_inference():
    report = setup.check_setup(ROOT, profile="ui")
    assert report["status"] == "setup_ready"
    assert report["real_inference"] == "NOT_RUN"
    assert report["real_backend"] == "UNVERIFIED"
    assert report["real_artifacts"] == "NOT_REQUESTED"


@pytest.mark.parametrize("missing", ["numpy", "streamlit"])
def test_missing_dependency_produces_action_without_install_or_model_load(monkeypatch, missing):
    original = setup.importlib.import_module

    def import_module(name):
        assert "ultralytics" not in name and "torch" not in name
        if name == missing:
            raise ModuleNotFoundError(f"No module named '{missing}'")
        return original(name)

    monkeypatch.setattr(setup.importlib, "import_module", import_module)
    report = setup.check_setup(ROOT, profile="ui")
    assert report["status"] == "blocked"
    failure = next(c for c in report["checks"] if c["name"] == missing)
    assert not failure["passed"]
    assert "pip install -e" in failure["action"]
    assert str(ROOT) in failure["action"]


def test_occupied_port_is_diagnosed_and_existing_listener_is_preserved():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        report = setup.check_setup(ROOT, profile="cli", port=port)
        check = next(c for c in report["checks"] if c["name"] == "ui_port")
        assert not check["passed"]
        assert str(port) in check["detail"]
        assert report["status"] == "blocked"
        assert listener.getsockname()[1] == port


def test_missing_source_and_bad_fake_config_are_reported(tmp_path):
    report = setup.check_setup(tmp_path, profile="cli")
    assert report["status"] == "blocked"
    assert not next(c for c in report["checks"] if c["name"] == "source_files")["passed"]
    assert not next(c for c in report["checks"] if c["name"] == "fake_config")["passed"]


def test_broken_dependency_metadata_blocks_setup(monkeypatch):
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="fixture dependency conflict", stderr=""
    ))
    report = setup.check_setup(ROOT, profile="cli")
    assert report["status"] == "blocked"
    assert not next(c for c in report["checks"] if c["name"] == "pip_check")["passed"]


@pytest.mark.parametrize("distribution,version", [
    ("streamlit", "1.20.0"), ("streamlit", "2.0.0"), ("numpy", "1.26.0"), ("Pillow", "13.0.0"),
])
def test_direct_profile_requirements_reject_wrong_versions_even_if_pip_check_passes(
    monkeypatch, distribution, version
):
    original = setup.importlib.metadata.version
    monkeypatch.setattr(setup.importlib.metadata, "version",
                        lambda name: version if name == distribution else original(name))
    report = setup.check_setup(ROOT, profile="ui")
    assert report["status"] == "blocked"
    assert not next(c for c in report["checks"] if c["name"] == distribution)["passed"]


def test_real_hash_check_is_not_model_execution_and_failure_never_falls_back(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"STUB ONLY, not a checkpoint")
    config = json.loads((ROOT / "configs/fake_app.json").read_bytes())
    config.update(mode="real", checkpoint_path="best.pt",
                  checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  model_origin="STUB_ONLY", nms_iou_threshold=0.5, imgsz=640)
    source = tmp_path / "real.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    original = setup.importlib.import_module

    def guarded_import(name):
        assert "ultralytics" not in name and "torch" not in name
        return original(name)

    monkeypatch.setattr(setup.importlib, "import_module", guarded_import)
    report = setup.check_setup(ROOT, profile="cli", real_config=source)
    assert report["status"] == "setup_ready"
    assert report["real_artifacts"] == "HASH_VALIDATED"
    assert report["real_inference"] == "NOT_RUN"
    weights.write_bytes(b"changed STUB")
    invalid = setup.check_setup(ROOT, profile="cli", real_config=source)
    assert invalid["status"] == "blocked"
    assert invalid["real_artifacts"] == "INVALID"
    assert invalid["real_inference"] == "NOT_RUN"
    missing = tmp_path / "missing.json"
    assert setup.check_setup(ROOT, profile="cli", real_config=missing)["status"] == "blocked"
