"""Portable Windows reserved-name checks for Python 3.11–3.13 on all OSes."""

import importlib.util
import io
import warnings
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("c_release_portable", ROOT / "scripts/package_c_release.py")
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
SHA = "a" * 40


@pytest.mark.parametrize("part", [
    "CON", "CON.py", "con.txt", "CON .txt", "PRN.md", "AUX.txt", "NUL.PY",
    "COM1.py", "COM9.md", "com2.py", "LPT1.txt", "lpt9.py",
    "CONIN$.md", "CONOUT$.txt", "COM¹.txt", "com².py", "LPT³.md",
])
def test_windows_reserved_basename_rejected_on_every_platform(part):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert not release.allowed("src/" + part)
        assert release._reserved_windows_component(part)


@pytest.mark.parametrize("name", [
    "src/console.py", "src/com10.py", "src/lpt0.py", "src/unreserved.py",
    "app/streamlit_app.py", "Start Jerry-C.sh", "Start Jerry-C.cmd",
])
def test_safe_paths_still_accepted(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert release.allowed(name)


def test_shell_launcher_is_required_in_source_manifest(tmp_path):
    sources = {name: b"synthetic packaging fixture\n" for name in release.REQUIRED}
    sources["Start Jerry-C.sh"] = (ROOT / "Start Jerry-C.sh").read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        first = release.make_archive(SHA, sources)
        assert first == release.make_archive(SHA, dict(reversed(list(sources.items()))))
        target = tmp_path / "source.zip"
        target.write_bytes(first)
        manifest = release.verify_release(target)
    assert manifest["source_commit"] == SHA
    assert manifest["scope"] == "C_SOURCE_ONLY"
    assert manifest["real_inference"] == "NOT_RUN"
    assert manifest["final_submission_ready"] is False
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.read("Start Jerry-C.sh").startswith(b"#!/usr/bin/env bash")
        assert archive.getinfo("Start Jerry-C.sh").external_attr >> 16 & 0o111


def test_release_rejects_laundered_reserved_paths(tmp_path):
    sources = {name: b"fixture\n" for name in release.REQUIRED}
    bad = "src/LPT1.py"
    with pytest.raises(ValueError, match="excluded"):
        release.make_archive(SHA, {**sources, bad: b"bad"})
    valid = release.make_archive(SHA, sources)
    with zipfile.ZipFile(io.BytesIO(valid)) as original:
        data = {name: original.read(name) for name in original.namelist()}
    data[bad] = b"bad"
    path = tmp_path / "tampered.zip"
    with zipfile.ZipFile(path, "w") as out:
        for name, bytes_ in data.items():
            out.writestr(name, bytes_)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="allowed"):
            release.verify_release(path)
