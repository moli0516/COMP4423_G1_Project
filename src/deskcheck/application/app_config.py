"""Explicit, source-bound configuration for C's synthetic integration."""

import hashlib
import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

CLASS_NAMES = ("bottle", "cup", "keyboard", "mouse", "book", "scissors")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _parse(content: bytes) -> dict:
    if not isinstance(content, bytes):
        raise TypeError("Configuration source must be immutable bytes")
    raw = json.loads(content, object_pairs_hook=_unique_keys)
    expected = {
        "schema_version",
        "mode",
        "confidence_threshold",
        "movement_threshold",
        "rules_version",
        "class_names",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("Configuration must contain exactly the documented fake settings")
    if raw["schema_version"] != "1.0" or raw["mode"] != "fake":
        raise ValueError("Only schema 1.0, explicitly fake mode is implemented")
    if raw["class_names"] != list(CLASS_NAMES):
        raise ValueError("Incompatible class mapping")
    for name in ("confidence_threshold", "movement_threshold"):
        value = raw[name]
        if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite within [0, 1]")
    if not isinstance(raw["rules_version"], str) or not raw["rules_version"].strip():
        raise ValueError("rules_version must be nonempty")
    return raw


@dataclass(frozen=True)
class AppConfig:
    confidence_threshold: float
    movement_threshold: float
    rules_version: str
    config_sha256: str
    _source_bytes: bytes = field(repr=False)
    mode: str = "fake"

    def __post_init__(self) -> None:
        raw = _parse(self._source_bytes)
        if hashlib.sha256(self._source_bytes).hexdigest() != self.config_sha256:
            raise ValueError("Configuration hash disagrees with its source")
        for name in ("confidence_threshold", "movement_threshold", "rules_version", "mode"):
            value = getattr(self, name)
            if type(value) is not type(raw[name]) or value != raw[name]:
                raise ValueError("Effective configuration disagrees with its hashed source")

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        content = Path(path).read_bytes()
        raw = _parse(content)
        return cls(
            raw["confidence_threshold"],
            raw["movement_threshold"],
            raw["rules_version"],
            hashlib.sha256(content).hexdigest(),
            content,
        )
