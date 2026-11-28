"""Round-trip tests for ModelArtifact with post-hoc provenance fields."""

from __future__ import annotations

from pathlib import Path

from deskcheck.training.model_artifact import ModelArtifact


def _base_artifact() -> ModelArtifact:
    return ModelArtifact(
        experiment_id="E0",
        run_dir=Path("/tmp/run"),
        checkpoint_path=Path("/tmp/run/best.pt"),
        checkpoint_sha256="a" * 64,
        architecture="yolov8n",
        pretrained_weights="yolov8n.pt",
        class_names=("bottle", "cup", "keyboard", "mouse", "book", "scissors"),
        image_size=640,
        dataset_yaml="/tmp/data/dataset.yaml",
        dataset_release="v1",
        split_seed=42,
        dataset_yaml_sha256="b" * 64,
        epochs_requested=100,
        epochs_completed=84,
        selected_epoch=64,
        checkpoint_selection="best_mAP50_95",
        batch_size=16,
        optimizer="SGD",
        learning_rate=0.01,
        weight_decay=0.0005,
        seed=42,
        device="0",
        confidence_threshold=0.20,
        nms_iou_threshold=0.35,
        ultralytics_version="8.4.215",
        torch_version="2.11.0+cu128",
        python_version="3.12.3",
    )


def test_roundtrip_without_provenance(tmp_path: Path) -> None:
    """Base artifact without provenance fields still round-trips."""
    a = _base_artifact()
    path = tmp_path / "artifact.json"
    a.save(path)
    loaded = ModelArtifact.load(path)
    assert loaded.experiment_id == "E0"
    assert loaded.resolved_config_at_training is None
    assert loaded.inference_at_training is None
    assert loaded.frozen_note is None


def test_roundtrip_with_provenance(tmp_path: Path) -> None:
    """Artifact with post-hoc provenance fields round-trips."""
    a = _base_artifact()
    from dataclasses import replace
    a2 = replace(
        a,
        resolved_config_at_training={"inference": {"confidence_threshold": 0.25}},
        inference_at_training={"confidence_threshold": 0.25, "nms_iou_threshold": 0.45},
        frozen_note="Frozen after validation-only tuning on 2026-11-20",
        resolved_config_note="resolved_config_at_training preserves pre-tuning values",
        post_hoc_annotation={
            "applied_at": "2026-11-24T18:00:00Z",
            "applied_by": "scripts/finalize_artifact.py",
            "reason": "Record frozen inference settings separately from training-time settings",
        },
    )
    path = tmp_path / "artifact.json"
    a2.save(path)
    loaded = ModelArtifact.load(path)
    assert loaded.resolved_config_at_training == {"inference": {"confidence_threshold": 0.25}}
    assert loaded.inference_at_training["nms_iou_threshold"] == 0.45
    assert "Frozen" in loaded.frozen_note
    assert loaded.post_hoc_annotation["applied_by"] == "scripts/finalize_artifact.py"
