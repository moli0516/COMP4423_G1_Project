"""Owner A: shared experiment-config loader/validator.

Reads a JSON experiment config, optionally merging an "inherit" base, validates
required keys and value ranges, and returns an immutable resolved config.

Does NOT import torch/ultralytics. Detector-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_TOP_LEVEL = {
    "experiment_id", "owner", "hypothesis", "baseline_reference",
    "changed_factor", "model", "data", "preprocessing", "augmentation",
    "training", "validation", "inference", "outputs", "notes",
}
ALLOWED_MODEL = {"architecture", "pretrained_weights", "trainable_layers"}
ALLOWED_DATA = {"dataset_yaml", "split_seed", "dataset_release", "class_map_version"}
ALLOWED_PREPROCESSING = {"image_size", "color_mode", "letterbox"}
ALLOWED_AUGMENTATION = {"hsv_h", "hsv_s", "hsv_v", "fliplr", "mosaic", "scale"}
ALLOWED_TRAINING = {
    "epochs", "batch_size", "optimizer", "lr0", "weight_decay",
    "patience", "seed", "device", "notes",
}
ALLOWED_VALIDATION = {"split", "checkpoint_selection"}
ALLOWED_INFERENCE = {"confidence_threshold", "nms_iou_threshold"}
ALLOWED_OUTPUTS = {"run_dir"}

INHERIT = "inherit"


@dataclass(frozen=True)
class TrainingConfig:
    experiment_id: str
    owner: str
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    augmentation: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    resolved_from_baseline: bool = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"config {path}: invalid JSON: {exc}") from exc


def _check_unknown_keys(name: str, block: dict[str, Any], allowed: set[str]) -> None:
    extra = set(block) - allowed
    if extra:
        raise ValueError(f"config block {name!r}: unknown keys {sorted(extra)}")


def _merge_inherit(child: Any, base: Any) -> Any:
    """Recursively merge child over base; 'inherit' means use base value."""
    if child == INHERIT:
        return base
    if isinstance(child, dict) and isinstance(base, dict):
        merged = dict(base)
        for key, value in child.items():
            merged[key] = _merge_inherit(value, base.get(key))
        return merged
    return child


def _validate_ranges(cfg: TrainingConfig) -> None:
    pre = cfg.preprocessing
    _require(isinstance(pre.get("image_size"), int) and pre["image_size"] > 0,
             "preprocessing.image_size must be a positive int")
    _require(pre.get("color_mode") == "RGB",
             "preprocessing.color_mode must be 'RGB'")
    _require(isinstance(pre.get("letterbox"), bool),
             "preprocessing.letterbox must be a bool")

    tr = cfg.training
    _require(isinstance(tr.get("epochs"), int) and tr["epochs"] > 0,
             "training.epochs must be a positive int")
    _require(isinstance(tr.get("batch_size"), int) and tr["batch_size"] > 0,
             "training.batch_size must be a positive int")
    _require(isinstance(tr.get("seed"), int),
             "training.seed must be an int")
    _require(0 < tr.get("lr0", -1) <= 1,
             "training.lr0 must be in (0, 1]")
    _require(tr.get("weight_decay", -1) >= 0,
             "training.weight_decay must be >= 0")

    aug = cfg.augmentation
    for key in ALLOWED_AUGMENTATION:
        if key in aug:
            _require(isinstance(aug[key], (int, float)) and aug[key] >= 0,
                     f"augmentation.{key} must be a non-negative number")

    inf = cfg.inference
    _require(0 <= inf.get("confidence_threshold", -1) <= 1,
             "inference.confidence_threshold must be in [0, 1]")
    _require(0 <= inf.get("nms_iou_threshold", -1) <= 1,
             "inference.nms_iou_threshold must be in [0, 1]")

    _require(cfg.validation.get("split") == "val",
             "validation.split must be 'val'; test evaluation is a separate "
             "execution path (no test-based model selection)")


def load(path: Path, baseline_path: Path | None = None) -> TrainingConfig:
    """Load, resolve inherit-keys against baseline, and validate."""
    raw = _read_json(path)

    _require("experiment_id" in raw, "config: missing 'experiment_id'")
    _require("owner" in raw, "config: missing 'owner'")
    _check_unknown_keys("top-level", raw, ALLOWED_TOP_LEVEL)

    baseline: dict[str, Any] | None = None
    if baseline_path is not None:
        baseline = _read_json(baseline_path)
        _check_unknown_keys("baseline top-level", baseline, ALLOWED_TOP_LEVEL)

    resolved: dict[str, Any] = dict(raw)
    if baseline is not None:
        for block_name, allowed in (
            ("model", ALLOWED_MODEL),
            ("data", ALLOWED_DATA),
            ("preprocessing", ALLOWED_PREPROCESSING),
            ("augmentation", ALLOWED_AUGMENTATION),
            ("training", ALLOWED_TRAINING),
            ("validation", ALLOWED_VALIDATION),
            ("inference", ALLOWED_INFERENCE),
            ("outputs", ALLOWED_OUTPUTS),
        ):
            if block_name not in resolved:
                resolved[block_name] = baseline.get(block_name, {})
            else:
                resolved[block_name] = _merge_inherit(
                    resolved[block_name], baseline.get(block_name, {})
                )
    else:
        for block_name in (
            "model", "data", "preprocessing", "augmentation",
            "training", "validation", "inference", "outputs",
        ):
            resolved.setdefault(block_name, {})

    for block_name, allowed in (
        ("model", ALLOWED_MODEL),
        ("data", ALLOWED_DATA),
        ("preprocessing", ALLOWED_PREPROCESSING),
        ("augmentation", ALLOWED_AUGMENTATION),
        ("training", ALLOWED_TRAINING),
        ("validation", ALLOWED_VALIDATION),
        ("inference", ALLOWED_INFERENCE),
        ("outputs", ALLOWED_OUTPUTS),
    ):
        block = resolved.get(block_name)
        _require(isinstance(block, dict),
                 f"config block {block_name!r} must be a JSON object")
        _check_unknown_keys(block_name, block, allowed)

    cfg = TrainingConfig(
        experiment_id=raw["experiment_id"],
        owner=raw["owner"],
        model=resolved["model"],
        data=resolved["data"],
        preprocessing=resolved["preprocessing"],
        augmentation=resolved["augmentation"],
        training=resolved["training"],
        validation=resolved["validation"],
        inference=resolved["inference"],
        outputs=resolved["outputs"],
        resolved_from_baseline=baseline is not None,
    )
    _validate_ranges(cfg)
    return cfg


def to_dict(cfg: TrainingConfig) -> dict[str, Any]:
    """Serialize the resolved config for logging/artifact metadata."""
    return {
        "experiment_id": cfg.experiment_id,
        "owner": cfg.owner,
        "resolved_from_baseline": cfg.resolved_from_baseline,
        "model": cfg.model,
        "data": cfg.data,
        "preprocessing": cfg.preprocessing,
        "augmentation": cfg.augmentation,
        "training": cfg.training,
        "validation": cfg.validation,
        "inference": cfg.inference,
        "outputs": cfg.outputs,
    }
