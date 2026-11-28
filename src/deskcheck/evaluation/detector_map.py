"""Detector mAP evaluation on an isolated copy of the dataset.

The module owns provenance hashing, dataset copying and metric validation; it
never imports a vision backend. Callers inject `validator`, which runs backend
validation on the copied dataset and returns raw metrics for one split.

`dataset_yaml_sha256` identifies the *input* dataset YAML. The evaluated copy
lives at `copy_dir`, where its `path:` field is rewritten to point at that copy
so backend validation cannot reach the original images.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_CLASS_NAMES: tuple[str, ...] = ("bottle", "cup", "keyboard", "mouse", "book", "scissors")
SPLITS: tuple[str, ...] = ("val", "test")

_PATH_PATTERN = re.compile(r"^path\s*:\s*(.*)$", flags=re.MULTILINE)
_YAML_SPECIAL = set(':#{}[]&*!|>\'"%@`,')
_MISSING = object()


def evaluate_detector_map(
    *,
    weights: str | Path,
    dataset_yaml: str | Path,
    split: str,
    output: str | Path,
    copy_dir: str | Path,
    validator: Callable[..., Mapping[str, object]],
    confirm_final_test: bool = False,
) -> dict:
    """Copy the dataset, validate against the copy, and write one mAP JSON report.

    The validator is called as ``validator(weights=..., dataset_yaml=...,
    split=...)`` with the copied YAML and must return a mapping containing
    map50, map50_95, class_names, per_class (name -> ap50/ap50_95),
    backend_name and backend_version. Missing, non-finite or misnamed values
    are rejected; nothing is written unless every field is present and valid.
    """
    weights_path = Path(weights)
    dataset_path = Path(dataset_yaml)
    output_path = Path(output)
    copy_path = Path(copy_dir)

    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    if split == "test" and not confirm_final_test:
        raise ValueError(
            "split=test is the held-out final evaluation and requires "
            "confirm_final_test=True (CLI: --confirm-final-test)"
        )
    if not callable(validator):
        raise TypeError("validator must be callable")
    for label, path in (("output", output_path), ("copy-dir", copy_path)):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing {label}: {path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset yaml not found: {dataset_path}")

    weights_sha256 = _sha256_file(weights_path)
    dataset_yaml_sha256 = _sha256_file(dataset_path)
    copied_yaml = _copy_dataset(dataset_path, copy_path)

    raw = validator(weights=weights_path, dataset_yaml=copied_yaml, split=split)
    report = _build_report(
        raw,
        split=split,
        weights_sha256=weights_sha256,
        dataset_yaml_sha256=dataset_yaml_sha256,
    )
    _write_json(output_path, report)
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _yaml_scalar(value: str) -> str:
    plain = value and value == value.strip() and not any(ch in _YAML_SPECIAL for ch in value)
    if plain:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _path_field(text: str) -> str | None:
    match = _PATH_PATTERN.search(text)
    if match is None:
        return None
    value = _strip_quotes(match.group(1).strip())
    if not value:
        raise ValueError("dataset yaml has an empty 'path:' field")
    return value


def _dataset_root(dataset_yaml: Path, text: str) -> Path:
    value = _path_field(text)
    root = dataset_yaml.parent if value is None else Path(value)
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    return root


def _rewrite_path_field(text: str, target: Path) -> str:
    # Without a 'path:' field, relative splits already resolve inside the yaml's directory.
    if _path_field(text) is None:
        return text
    return _PATH_PATTERN.sub(lambda _match: f"path: {_yaml_scalar(str(target))}", text, count=1)


def _copy_dataset(dataset_yaml: Path, copy_dir: Path) -> Path:
    text = dataset_yaml.read_text(encoding="utf-8")
    root = _dataset_root(dataset_yaml, text)
    target = copy_dir.resolve()
    if target == root or root in target.parents or target in root.parents:
        raise ValueError(f"copy-dir must be outside the dataset root {root}: {copy_dir}")

    copy_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{copy_dir.name}-", dir=copy_dir.parent))
    try:
        tree = staging / "dataset"
        shutil.copytree(root, tree, symlinks=False)
        (tree / dataset_yaml.name).write_text(_rewrite_path_field(text, target), encoding="utf-8")
        if copy_dir.exists() or copy_dir.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing copy-dir: {copy_dir}")
        tree.rename(copy_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return copy_dir / dataset_yaml.name


def _finite(value: object, label: str) -> float:
    if value is _MISSING or value is None:
        raise ValueError(f"backend result is missing metric: {label}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {number!r}")
    return number


def _text(value: object, label: str) -> str:
    if value is _MISSING or value is None:
        raise ValueError(f"backend result is missing field: {label}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _build_report(
    raw: object,
    *,
    split: str,
    weights_sha256: str,
    dataset_yaml_sha256: str,
) -> dict:
    if not isinstance(raw, Mapping):
        raise TypeError("validator must return a mapping of backend metrics")

    names = raw.get("class_names", _MISSING)
    if names is _MISSING:
        raise ValueError("backend result is missing metric: class_names")
    if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
        raise TypeError("class_names must be a sequence of strings")
    if tuple(names) != EXPECTED_CLASS_NAMES:
        raise ValueError(
            "backend class list does not match the approved detector classes: "
            f"expected {list(EXPECTED_CLASS_NAMES)}, got {list(names)}"
        )

    raw_per_class = raw.get("per_class", _MISSING)
    if raw_per_class is _MISSING:
        raise ValueError("backend result is missing metric: per_class")
    if not isinstance(raw_per_class, Mapping):
        raise TypeError("per_class must be a mapping of class name to AP metrics")
    unknown = sorted(str(key) for key in raw_per_class if key not in EXPECTED_CLASS_NAMES)
    if unknown:
        raise ValueError(f"per_class contains classes outside the approved list: {unknown}")

    per_class = {}
    for name in EXPECTED_CLASS_NAMES:
        entry = raw_per_class.get(name, _MISSING)
        if entry is _MISSING:
            raise ValueError(f"per_class is missing class: {name}")
        if not isinstance(entry, Mapping):
            raise TypeError(f"per_class[{name}] must be a mapping of AP metrics")
        per_class[name] = {
            "ap50": _finite(entry.get("ap50", _MISSING), f"per_class[{name}].ap50"),
            "ap50_95": _finite(entry.get("ap50_95", _MISSING), f"per_class[{name}].ap50_95"),
        }

    return {
        "map50": _finite(raw.get("map50", _MISSING), "map50"),
        "map50_95": _finite(raw.get("map50_95", _MISSING), "map50_95"),
        "per_class": per_class,
        "class_names": list(EXPECTED_CLASS_NAMES),
        "split": split,
        "weights_sha256": weights_sha256,
        "dataset_yaml_sha256": dataset_yaml_sha256,
        "backend_name": _text(raw.get("backend_name", _MISSING), "backend_name"),
        "backend_version": _text(raw.get("backend_version", _MISSING), "backend_version"),
        "utc_timestamp": datetime.now(UTC).isoformat(),
    }


def _write_json(output: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{output.name}-", dir=output.parent)
    os.close(descriptor)
    staging = Path(temp_name)
    try:
        staging.write_text(text, encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        staging.replace(output)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
