"""Strict config for externally supplied trained-model inference (no auto-download)."""

import hashlib
import json
from dataclasses import dataclass
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
    mode: str = "real"

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
        actual = sha256_file(checkpoint)
        if actual.lower() != digest.lower():
            raise ValueError("Checkpoint hash mismatch: refusal prevents unverified inference")
        return cls(
            checkpoint_path=checkpoint,
            checkpoint_sha256=actual,
            config_sha256=hashlib.sha256(data).hexdigest(),
            model_origin=raw["model_origin"],
            confidence_threshold=raw["confidence_threshold"],
            movement_threshold=raw["movement_threshold"],
            nms_iou_threshold=raw["nms_iou_threshold"],
            imgsz=raw["imgsz"],
            rules_version=raw["rules_version"],
        )
