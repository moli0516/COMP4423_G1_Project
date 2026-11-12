"""Reproducible, integrity-checked source packaging; synthetic release only."""

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("c_release", ROOT / "scripts/package_c_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
SHA = "a" * 40


def payloads():
    return {name: b"synthetic packaging fixture\n" for name in release.REQUIRED}


def test_archive_is_repeatable_complete_and_offline_verifiable(tmp_path):
    first = release.make_archive(SHA, payloads())
    assert first == release.make_archive(SHA, dict(reversed(list(payloads().items()))))
    path = tmp_path / "source.zip"
    path.write_bytes(first)
    manifest = release.verify_release(path)
    assert manifest["source_commit"] == SHA
    assert manifest["scope"] == "C_SOURCE_ONLY"
    assert manifest["final_submission_ready"] is False
    assert manifest["real_inference"] == "NOT_RUN"


@pytest.mark.parametrize("name", [
    "../leak.py", "/absolute.py", "src/../leak.py", "src\\leak.py", "src//leak.py",
    "src/./leak.py", ".env", ".venv/runtime.py", "data/photo.jpg", "artifacts/best.pt", "",
    "src/NUL.py", "src/CON.py", "src/LPT1.py", "src/trailing./file.py", "src/trailing /file.py",
    "src/bad?.py",
])
def test_private_binary_and_unsafe_paths_cannot_enter_source_archive(name):
    assert not release.allowed(name)
    with pytest.raises(ValueError):
        release.make_archive(SHA, {**payloads(), name: b"excluded"})


@pytest.mark.parametrize("change", ["content", "extra", "missing", "manifest", "symlink"])
def test_archive_tampering_is_rejected(tmp_path, change):
    raw = release.make_archive(SHA, payloads())
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    name = "scripts/run_c_demo.py"
    if change == "content":
        entries[name] = b"altered"
    elif change == "extra":
        entries["unlisted.py"] = b"altered"
    elif change == "missing":
        entries.pop(name)
    elif change == "manifest":
        entries[release.MANIFEST] = json.dumps([]).encode()
    path = tmp_path / "changed.zip"
    with zipfile.ZipFile(path, "w") as target:
        for key, data in entries.items():
            info = zipfile.ZipInfo(key)
            if change == "symlink" and key == name:
                info.create_system = 3
                info.external_attr = 0o120777 << 16
            target.writestr(info, data)
    with pytest.raises(ValueError):
        release.verify_release(path)


def test_dirty_worktree_blocks_build_before_reading_sources(monkeypatch):
    calls = []

    def git_bytes(root, *args):
        calls.append(args)
        return b"Jerry-C\n" if args[0] == "branch" else b" M src/owned.py\n"

    monkeypatch.setattr(release, "git_bytes", git_bytes)
    with pytest.raises(ValueError, match="dirty"):
        release.committed_sources(ROOT)
    assert [call[0] for call in calls] == ["branch", "status"]


def test_existing_archive_is_preserved_byte_for_byte(tmp_path, monkeypatch):
    path = tmp_path / "preserved.zip"
    path.write_bytes(b"existing user artifact")
    monkeypatch.setattr(release.sys, "argv", ["package_c_release.py", "--output", str(path)])
    assert release.main() == 1
    assert path.read_bytes() == b"existing user artifact"


@pytest.mark.parametrize("names", [
    ["src/Foo.py", "src/foo.py"], ["src/file.py", "src/file.py/child.py"],
])
def test_windows_path_aliases_cannot_overwrite_each_other(names):
    with pytest.raises(ValueError, match="(?i)colli"):
        release.make_archive(SHA, {**payloads(), **dict.fromkeys(names, b"fixture")})
