"""Unit tests for training_config.load — validation + inherit resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deskcheck.training.training_config import (
    TrainingConfig,
    load,
    to_dict,
)


def _baseline(tmp_path: Path) -> Path:
    baseline = {
        "experiment_id": "E0", "owner": "A",
        "baseline_reference": "pretrained:yolov8n",
        "model": {"architecture": "yolov8n", "pretrained_weights": "yolov8n.pt",
                  "trainable_layers": "all"},
        "data": {"dataset_yaml": "x.yaml", "split_seed": 42,
                 "dataset_release": "v1", "class_map_version": "1.0"},
        "preprocessing": {"image_size": 640, "color_mode": "RGB", "letterbox": True},
        "augmentation": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                         "fliplr": 0.5, "mosaic": 1.0, "scale": 0.5},
        "training": {"epochs": 100, "batch_size": 16, "optimizer": "SGD",
                     "lr0": 0.01, "weight_decay": 0.0005, "patience": 20,
                     "seed": 42, "device": "auto"},
        "validation": {"split": "val", "checkpoint_selection": "best_mAP50_95"},
        "inference": {"confidence_threshold": 0.25, "nms_iou_threshold": 0.45},
        "outputs": {"run_dir": "runs/E0"},
        "notes": "test baseline",
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    return path


def _child(tmp_path: Path, overrides: dict) -> Path:
    path = tmp_path / "child.json"
    path.write_text(json.dumps(overrides), encoding="utf-8")
    return path


# ---------- happy paths ----------


def test_load_baseline_standalone(tmp_path: Path) -> None:
    cfg = load(_baseline(tmp_path))
    assert isinstance(cfg, TrainingConfig)
    assert cfg.experiment_id == "E0"
    assert cfg.owner == "A"
    assert cfg.preprocessing["image_size"] == 640
    assert cfg.training["epochs"] == 100
    assert cfg.resolved_from_baseline is False


def test_load_child_with_inherit(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    child = _child(tmp_path, {
        "experiment_id": "E1", "owner": "B",
        "baseline_reference": "baseline.json",
        "changed_factor": "augmentation",
        "augmentation": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                         "fliplr": 0.0, "mosaic": 0.0, "scale": 0.0},
        "training": "inherit",
        "preprocessing": "inherit",
        "model": "inherit",
        "data": "inherit",
        "validation": "inherit",
        "inference": "inherit",
        "outputs": {"run_dir": "runs/E1"},
    })
    cfg = load(child, baseline_path=base)
    assert cfg.experiment_id == "E1"
    assert cfg.owner == "B"
    # augmentation changed
    assert cfg.augmentation["mosaic"] == 0.0
    # other blocks inherited
    assert cfg.training["epochs"] == 100
    assert cfg.training["batch_size"] == 16
    assert cfg.preprocessing["image_size"] == 640
    assert cfg.model["architecture"] == "yolov8n"
    assert cfg.resolved_from_baseline is True


def test_load_partial_inherit(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    child = _child(tmp_path, {
        "experiment_id": "E2", "owner": "C",
        "baseline_reference": "baseline.json",
        "changed_factor": "preprocessing.image_size",
        "preprocessing": {"image_size": 960},
        "training": {"batch_size": 8},
        "model": "inherit",
        "data": "inherit",
        "augmentation": "inherit",
        "validation": "inherit",
        "inference": "inherit",
        "outputs": {"run_dir": "runs/E2"},
    })
    cfg = load(child, baseline_path=base)
    assert cfg.preprocessing["image_size"] == 960
    assert cfg.preprocessing["color_mode"] == "RGB"  # inherited from base
    assert cfg.training["batch_size"] == 8
    assert cfg.training["epochs"] == 100  # inherited


def test_to_dict_roundtrip(tmp_path: Path) -> None:
    cfg = load(_baseline(tmp_path))
    d = to_dict(cfg)
    assert d["experiment_id"] == "E0"
    assert d["training"]["epochs"] == 100
    assert "resolved_from_baseline" in d


# ---------- error paths ----------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load(path)


def test_missing_experiment_id_raises(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"owner": "A"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'experiment_id'"):
        load(path, baseline_path=base)


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["rogue_field"] = "surprise"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load(path, baseline_path=base)


def test_unknown_block_key_raises(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["training"]["made_up_lr"] = 0.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load(path, baseline_path=base)


def test_invalid_image_size_raises(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["preprocessing"]["image_size"] = -100
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="image_size"):
        load(path, baseline_path=base)


def test_invalid_confidence_threshold_raises(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["inference"]["confidence_threshold"] = 1.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="confidence_threshold"):
        load(path, baseline_path=base)


def test_invalid_validation_split_raises(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["validation"]["split"] = "train"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation.split"):
        load(path, baseline_path=base)


def test_inherit_without_baseline_raises(tmp_path: Path) -> None:
    child = _child(tmp_path, {
        "experiment_id": "E1", "owner": "B",
        "training": "inherit",
    })
    with pytest.raises(ValueError, match="must be a JSON object"):
        load(child, baseline_path=None)


def test_missing_outputs_block_inherits_from_baseline(tmp_path: Path) -> None:
    # If a block is absent from the child config, load() inherits it from the
    # baseline rather than defaulting to an empty dict. (Blocks marked
    # explicitly as "inherit" behave the same way; absent and "inherit" are
    # equivalent when a baseline is supplied.)
    base = _baseline(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    del payload["outputs"]
    path = tmp_path / "no_outputs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = load(path, baseline_path=base)
    assert cfg.outputs == {"run_dir": "runs/E0"}

def test_rejects_test_split_for_checkpoint_selection(tmp_path: Path) -> None:
    base = _baseline(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["validation"] = {
        "split": "test",
        "checkpoint_selection": "best_mAP50_95",
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation.split must be 'val'"):
        load(path, baseline_path=base)
