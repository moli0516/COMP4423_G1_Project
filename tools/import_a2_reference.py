#!/usr/bin/env python3
"""Fetch a pinned A2 reference snapshot. No production code is modified."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

OWNER = "moli0516"
REPO = "COMP4423_A2"
REVISION = "a9c473a1634ea3ad9be2bea45457a4b9dd53e8a4"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "references" / "a2_reuse" / "original"
SEEDS = [
    "scripts/update_images_csv.py", "scripts/split_dataset.py",
    "scripts/validate_dataset.py", "src/data/dataset_catalog.py",
    "src/data/constants.py", "src/utils/file_io.py",
    "src/utils/serialization.py", "src/experiments/polyu_lr_config.py",
    "src/preprocessing/image_preprocessor.py",
    "src/preprocessing/preprocessing_config.py",
]
PREFIXES = ("tests/splitting/", "tests/preprocessing/", "src/exceptions/")


def request(url):
    req = Request(url, headers={"User-Agent": "DeskCheck-A2-reference-importer"})
    with urlopen(req, timeout=45) as response:
        return response.read()


def main():
    if DEST.exists():
        raise SystemExit(f"Refusing existing destination: {DEST}")
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{REVISION}?recursive=1"
    listing = json.loads(request(url))
    if listing.get("truncated"):
        raise RuntimeError("GitHub tree is truncated; refusing an incomplete import")
    available = {e["path"] for e in listing["tree"] if e["type"] == "blob"}
    selected = set(SEEDS)
    selected.update(p for p in available if p.endswith(".py") and p.startswith(PREFIXES))
    missing = selected - available
    if missing:
        raise RuntimeError(f"Missing pinned source files: {sorted(missing)}")
    payloads = {}
    pending = set(selected)

    def add_module(parts):
        if not parts or parts[0] not in {"src", "tests", "scripts"}:
            return
        base = "/".join(parts)
        candidates = [base + ".py", base + "/__init__.py"]
        for candidate in candidates:
            if candidate in available and candidate not in payloads:
                pending.add(candidate)

    while pending:
        path = min(pending)
        pending.remove(path)
        if path in payloads:
            continue
        raw = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{REVISION}/{path}"
        data = request(raw)
        payloads[path] = data
        module = ast.parse(data, filename=path)
        package = list(PurePosixPath(path).parent.parts)
        for i in range(1, len(package) + 1):
            init = "/".join(package[:i]) + "/__init__.py"
            if init in available and init not in payloads:
                pending.add(init)
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    trim = node.level - 1
                    base = package[:len(package) - trim] if trim else package[:]
                else:
                    base = []
                if node.module:
                    base += node.module.split(".")
                add_module(base)
                for alias in node.names:
                    if alias.name != "*":
                        add_module(base + alias.name.split("."))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".a2-import-", dir=DEST.parent))
    try:
        entries = []
        for path, data in sorted(payloads.items()):
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entries.append({"source_path": path, "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {"repository": f"https://github.com/{OWNER}/{REPO}",
                    "revision": REVISION, "files": entries,
                    "limitations": "Static Python imports only. Inspect dynamic imports and non-Python fixtures manually. Not integration-tested."}
        (staging / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if DEST.exists():
            raise RuntimeError("Destination appeared during download; refusing overwrite")
        staging.rename(DEST)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Imported {len(payloads)} Python files at {REVISION} into {DEST}")
    print("Review manifest, test fixtures and license/course reuse rules before committing.")


if __name__ == "__main__":
    main()
