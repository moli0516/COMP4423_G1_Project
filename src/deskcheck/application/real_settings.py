"""Strict config for externally supplied trained-model inference (no auto-download)."""

import hashlib
import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from deskcheck.application.app_config import CLASS_NAMES

_FIELDS = {
    "schema_version",
    "mode",
    "checkpoint_path",
    "checkpoint_sha256",
    "model_origin",
    "confidence_threshold",
    "movement_threshold",
    "nms_iou_threshold",
    "imgsz",
    "rules_version",
    "class_names",
}


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate real config key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RealSettings:
    checkpoint_path: Path
    checkpoint_sha256: str
    config_sha256: str
    model_origin: str
    confidence_threshold: float
    movement_threshold: float
    nms_iou_threshold: float
    imgsz: int
    rules_version: str
    _source_bytes: bytes = field(repr=False)
    _source_path: Path = field(repr=False)
    mode: str = "real"

    def __post_init__(self) -> None:
        """Validate effective parameters against immutable configuration-source bytes.

        This is required even for direct dataclass construction and
        dataclasses.replace(), which bypasses load() but invokes __post_init__.
        """
        if type(self._source_bytes) is not bytes or not isinstance(self._source_path, Path):
            raise TypeError("RealSettings requires immutable source bytes and source path")
        if not self._source_path.is_absolute() or self._source_path != self._source_path.resolve():
            raise ValueError("Real config source path must be absolute and resolved")
        source_hash = hashlib.sha256(self._source_bytes).hexdigest()
        if self.config_sha256 != source_hash:
            raise ValueError("Effective config hash disagrees with the source bytes")
        raw = json.loads(self._source_bytes, object_pairs_hook=_no_duplicate_keys)
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise ValueError("Real config source must contain exactly documented fields")
        if raw["schema_version"] != "1.0" or raw["mode"] != "real":
            raise ValueError("Real config source must be schema 1.0 / real")
        if raw["class_names"] != list(CLASS_NAMES):
            raise ValueError("Real config source has an incompatible class map")
        for name in ("rules_version", "model_origin", "checkpoint_path"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ValueError(f"Real source {name} must be nonempty")
        for name in ("confidence_threshold", "movement_threshold", "nms_iou_threshold"):
            value = raw[name]
            if type(value) not in (float, int) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Real source {name} must be finite within [0,1]")
        if type(raw["imgsz"]) is not int or not 32 <= raw["imgsz"] <= 4096:
            raise ValueError("Real source imgsz must be an integer 32..4096")
        for name in (
            "model_origin",
            "confidence_threshold",
            "movement_threshold",
            "nms_iou_threshold",
            "imgsz",
            "rules_version",
            "mode",
        ):
            actual = getattr(self, name)
            required = raw[name]
            if type(actual) is not type(required) or actual != required:
                raise ValueError(f"Effective {name} disagrees with its hashed source")
        checkpoint = Path(raw["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = self._source_path.parent / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
            raise FileNotFoundError(f"Local .pt checkpoint is missing: {checkpoint}")
        if self.checkpoint_path != checkpoint:
            raise ValueError("Effective checkpoint path disagrees with hashed config source")
        raw_hash = raw["checkpoint_sha256"]
        if (
            not isinstance(raw_hash, str)
            or len(raw_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in raw_hash.lower())
        ):
            raise ValueError("Hashed config contains invalid checkpoint SHA256")
        if self.checkpoint_sha256 != raw_hash.lower():
            raise ValueError("Effective checkpoint hash disagrees with hashed config source")
        if sha256_file(checkpoint) != self.checkpoint_sha256:
            raise ValueError("Local checkpoint hash disagrees with declared source hash")

    @classmethod
    def load(cls, source: str | Path) -> "RealSettings":
        config_path = Path(source).resolve(strict=True)
        data = config_path.read_bytes()
        raw = json.loads(data, object_pairs_hook=_no_duplicate_keys)
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise ValueError("Real config requires exactly the documented fields")
        if raw["schema_version"] != "1.0" or raw["mode"] != "real":
            raise ValueError("Real config must explicitly select schema 1.0 / real mode")
        if raw["class_names"] != list(CLASS_NAMES):
            raise ValueError("Real config must match the six canonical classes in order")
        for key in ("rules_version", "model_origin", "checkpoint_path"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise ValueError(f"Real config requires a nonempty {key}")
        for key in ("confidence_threshold", "movement_threshold", "nms_iou_threshold"):
            val = raw[key]
            if type(val) not in (float, int) or not isfinite(val) or not 0 <= val <= 1:
                raise ValueError(f"{key} must be finite and in [0,1]")
        if type(raw["imgsz"]) is not int or not 32 <= raw["imgsz"] <= 4096:
            raise ValueError("imgsz must be an integer 32..4096")
        digest = raw["checkpoint_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest.lower())
        ):
            raise ValueError("checkpoint_sha256 must be 64 hex characters")
        checkpoint = Path(raw["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = config_path.parent / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
            raise FileNotFoundError(f"Local .pt checkpoint is missing: {checkpoint}")
        # __post_init__ verifies the full checkpoint digest once, including
        # for direct constructors and dataclasses.replace().
        return cls(
            checkpoint_path=checkpoint,
            checkpoint_sha256=digest.lower(),
            config_sha256=hashlib.sha256(data).hexdigest(),
            model_origin=raw["model_origin"],
            confidence_threshold=raw["confidence_threshold"],
            movement_threshold=raw["movement_threshold"],
            nms_iou_threshold=raw["nms_iou_threshold"],
            imgsz=raw["imgsz"],
            rules_version=raw["rules_version"],
            _source_bytes=data,
            _source_path=config_path,
        )
