"""Owner A: load approved pair annotations and split manifests.

Pure I/O + validation. Does not touch models or B/C code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
REQUIRED_PAIR_FIELDS = {
    "schema_version",
    "pair_id",
    "capture_group_id",
    "before_image_id",
    "after_image_id",
    "before_image_path",
    "after_image_path",
    "width",
    "height",
    "rules_version",
    "independently_captured_no_change",
    "manual_before_objects",
    "manual_after_objects",
    "correspondences",
    "events",
}
REQUIRED_OBJECT_FIELDS = {"object_id", "class_id", "bbox_xyxy"}
REQUIRED_EVENT_FIELDS = {"class_id", "status"}
VALID_STATUSES = {"added", "removed", "moved", "unchanged"}


@dataclass(frozen=True)
class AnnotatedPair:
    """One before/after pair with ground-truth annotations. No runtime images."""

    pair_id: str
    capture_group_id: str
    before_image_id: str
    after_image_id: str
    before_image_path: str
    after_image_path: str
    width: int
    height: int
    rules_version: str
    independently_captured_no_change: bool
    manual_before_objects: tuple[dict[str, Any], ...]
    manual_after_objects: tuple[dict[str, Any], ...]
    correspondences: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class SplitManifest:
    schema_version: str
    seed: int
    dataset_release: str
    splits: dict[str, tuple[str, ...]]
    coverage: dict[str, Any] = field(default_factory=dict)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_bool(value: Any, what: str) -> bool:
    """Reject truthy coercions such as bool('false') == True."""
    if isinstance(value, bool):
        return value
    raise ValueError(f"{what} must be an actual boolean, got {type(value).__name__}")


def _parse_pair(raw: dict[str, Any], line_no: int) -> AnnotatedPair:
    missing = REQUIRED_PAIR_FIELDS - raw.keys()
    _require(not missing, f"pair on line {line_no}: missing fields {sorted(missing)}")
    _require(raw["schema_version"] == SCHEMA_VERSION,
             f"pair on line {line_no}: schema_version must be {SCHEMA_VERSION}")
    for obj_list_name in ("manual_before_objects", "manual_after_objects"):
        for obj in raw[obj_list_name]:
            obj_missing = REQUIRED_OBJECT_FIELDS - obj.keys()
            _require(not obj_missing,
                     f"pair {raw['pair_id']} {obj_list_name}: missing {sorted(obj_missing)}")
    for ev in raw["events"]:
        ev_missing = REQUIRED_EVENT_FIELDS - ev.keys()
        _require(not ev_missing,
                 f"pair {raw['pair_id']} event: missing {sorted(ev_missing)}")
        _require(ev["status"] in VALID_STATUSES,
                 f"pair {raw['pair_id']}: invalid status {ev['status']!r}")
    _require(isinstance(raw["width"], int) and raw["width"] > 0,
             f"pair {raw['pair_id']}: width must be positive int")
    _require(isinstance(raw["height"], int) and raw["height"] > 0,
             f"pair {raw['pair_id']}: height must be positive int")
    return AnnotatedPair(
        pair_id=raw["pair_id"],
        capture_group_id=raw["capture_group_id"],
        before_image_id=raw["before_image_id"],
        after_image_id=raw["after_image_id"],
        before_image_path=raw["before_image_path"],
        after_image_path=raw["after_image_path"],
        width=raw["width"],
        height=raw["height"],
        rules_version=raw["rules_version"],
        independently_captured_no_change=_parse_bool(
            raw["independently_captured_no_change"],
            f"pair {raw['pair_id']!r} independently_captured_no_change",
        ),
        manual_before_objects=tuple(raw["manual_before_objects"]),
        manual_after_objects=tuple(raw["manual_after_objects"]),
        correspondences=tuple(raw["correspondences"]),
        events=tuple(raw["events"]),
    )


def load_annotations(path: Path) -> dict[str, AnnotatedPair]:
    """Read JSONL annotations. Reject duplicate pair_ids and malformed records."""
    if not path.exists():
        raise FileNotFoundError(f"annotations file not found: {path}")
    pairs: dict[str, AnnotatedPair] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"annotations line {line_no}: invalid JSON: {exc}") from exc
            pair = _parse_pair(raw, line_no)
            if pair.pair_id in pairs:
                raise ValueError(f"duplicate pair_id {pair.pair_id!r} at line {line_no}")
            pairs[pair.pair_id] = pair
    return pairs


def load_manifest(path: Path) -> SplitManifest:
    """Read split manifest JSON. Reject missing keys and unknown split names."""
    if not path.exists():
        raise FileNotFoundError(f"manifest file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema_version", "seed", "dataset_release", "splits"):
        _require(key in raw, f"manifest: missing {key!r}")
    splits_raw = raw["splits"]
    _require(set(splits_raw.keys()) == {"train", "val", "test"},
             "manifest: splits must contain exactly train, val, test")
    splits = {name: tuple(ids) for name, ids in splits_raw.items()}
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for pair_id in ids:
            if pair_id in seen:
                raise ValueError(
                    f"pair_id {pair_id!r} appears in both {seen[pair_id]!r} and {name!r}"
                )
            seen[pair_id] = name
    return SplitManifest(
        schema_version=raw["schema_version"],
        seed=int(raw["seed"]),
        dataset_release=raw["dataset_release"],
        splits=splits,
        coverage=raw.get("coverage", {}),
    )


def load_catalog(
    annotation_path: Path, manifest_path: Path
) -> tuple[dict[str, AnnotatedPair], SplitManifest]:
    """Load both files and cross-check that every manifest pair_id exists."""
    pairs = load_annotations(annotation_path)
    manifest = load_manifest(manifest_path)
    manifest_ids = {pid for ids in manifest.splits.values() for pid in ids}
    missing = manifest_ids - pairs.keys()
    if missing:
        raise ValueError(f"manifest references unknown pair_ids: {sorted(missing)}")
    return pairs, manifest
