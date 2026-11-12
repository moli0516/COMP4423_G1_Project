"""Build/verify a reproducible C source archive; not a complete course submission.

Only committed, explicitly allowed source paths are packaged. No venv, private
photos, checkpoint, dataset or student report is added. No model is loaded.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md", "pyproject.toml", "requirements.txt", "requirements-windows-py312.lock.txt",
    "Start Jerry-C.cmd", "STATUS_BOARD.md", ".gitignore", "LICENSE",
}
SOURCE_DIRS = {"src", "app", "scripts", "tests", "docs"}
SOURCE_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".toml"}
REQUIRED = {
    "pyproject.toml", "requirements.txt", "configs/fake_app.json", "app/streamlit_app.py",
    "scripts/check_c_setup.py", "scripts/run_c_demo.py", "scripts/package_c_release.py",
    "src/deskcheck/application/comparison_service.py",
}
MANIFEST = "RELEASE_MANIFEST.json"


def allowed(name: str) -> bool:
    if not isinstance(name, str) or not name:
        return False
    path = PurePosixPath(name)
    if (
        not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
        or path.as_posix() != name
        or any(
            part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
            or any(ord(char) < 32 or char in '<>"|?*' for char in part)
            for part in path.parts
        )
    ):
        return False
    return (
        name in ROOT_FILES or name == "configs/fake_app.json"
        or (path.parts[0] in SOURCE_DIRS and path.suffix in SOURCE_SUFFIXES)
    )


def validate_names(names) -> None:
    folded = {name.casefold() for name in names}
    if len(names) != len(folded):
        raise ValueError("Duplicate or case-colliding archive paths")
    for name in names:
        if any(parent.as_posix().casefold() in folded for parent in PurePosixPath(name).parents):
            raise ValueError("Archive file/directory path collision")


def git_bytes(root: Path, *args) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                            check=False, timeout=60)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def committed_sources(root: Path) -> tuple[str, dict[str, bytes]]:
    if git_bytes(root, "branch", "--show-current").strip() != b"Jerry-C":
        raise ValueError("Build only from the Jerry-C branch")
    if git_bytes(root, "status", "--porcelain=v1", "--untracked-files=normal").strip():
        raise ValueError("Source worktree is dirty; preserve changes and commit/review before release")
    commit = git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    names = git_bytes(root, "ls-tree", "-r", "--name-only", "-z", commit).decode("utf-8").split("\0")
    selected = sorted(name for name in names if name and allowed(name))
    archive = git_bytes(root, "archive", "--format=zip", commit, "--", *selected)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        payloads = {name: source.read(name) for name in selected}
        if any(source.getinfo(name).external_attr >> 16 & 0o170000 == 0o120000 for name in selected):
            raise ValueError("Symlink sources cannot be packaged")
    return commit, payloads


def make_archive(commit: str, payloads: dict[str, bytes]) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Source commit must be a full Git SHA")
    if not REQUIRED <= payloads.keys() or any(not allowed(name) for name in payloads):
        raise ValueError("Source inventory is incomplete or contains excluded paths")
    validate_names([*payloads, MANIFEST])
    manifest = {
        "format_version": "1.0", "scope": "C_SOURCE_ONLY", "source_commit": commit,
        "real_inference": "NOT_RUN", "final_submission_ready": False,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
        "limitations": ["Dataset, checkpoint, A/B approvals and individual reports not included.",
                        "Synthetic demo only; real runtime/accuracy not certified."],
    }
    entries = {**payloads, MANIFEST: (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return output.getvalue()


def verify_release(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        validate_names(names)
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError("Duplicate archive paths or missing release manifest")
        manifest = json.loads(archive.read(MANIFEST))
        if (
            not isinstance(manifest, dict)
            or manifest.get("format_version") != "1.0" or manifest.get("scope") != "C_SOURCE_ONLY"
            or manifest.get("real_inference") != "NOT_RUN"
            or manifest.get("final_submission_ready") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_commit", ""))
        ):
            raise ValueError("Invalid C source manifest")
        files = manifest.get("files")
        if not isinstance(files, dict) or not REQUIRED <= files.keys():
            raise ValueError("Incomplete C source manifest")
        if set(names) != files.keys() | {MANIFEST} or any(not allowed(name) for name in files):
            raise ValueError("Archive inventory disagrees with allowed source manifest")
        for name, expected in files.items():
            if archive.getinfo(name).external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError("Symlink archive entry")
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive integrity mismatch: {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Fresh C-source ZIP path (requires clean Git checkout)")
    mode.add_argument("--verify", type=Path, help="Read-only archive integrity check; no Git needed")
    args = parser.parse_args()
    try:
        if args.verify:
            manifest = verify_release(args.verify)
            print(f"Verified C_SOURCE_ONLY: {manifest['source_commit']}; {len(manifest['files'])} files")
        else:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(f"Refusing to overwrite release: {args.output}")
            commit, payloads = committed_sources(ROOT)
            content = make_archive(commit, payloads)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as target:
                target.write(content)
            manifest = verify_release(args.output)
            print(f"C_SOURCE_ONLY ready: {args.output.resolve()}; source={commit}; "
                  f"{len(manifest['files'])} files. Not a complete course submission.")
        return 0
    except (OSError, ValueError, TypeError, zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
        print(f"C release failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
