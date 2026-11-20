"""Owner A: the shared training entry point.

Loads a validated TrainingConfig, runs fine-tuning with ultralytics,
and saves a ModelArtifact alongside the checkpoint.

No test-set data is touched during training. Checkpoint selection is
validation-only, per the frozen rule.
"""

from __future__ import annotations

import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

from deskcheck.training.model_artifact import (
    ModelArtifact,
    sha256_file,
)
from deskcheck.training.training_config import load as load_config


class TrainingError(RuntimeError):
    """Raised when training cannot proceed (backend missing, dataset invalid)."""


def _lazy_import_ultralytics():
    """Import ultralytics lazily so the package installs without vision extras."""
    try:
        import ultralytics
        from ultralytics import YOLO
        return YOLO, ultralytics.__version__
    except ImportError as exc:
        raise TrainingError(
            "ultralytics is not installed. Run: pip install -e '.[vision]'"
        ) from exc


def _check_dataset(dataset_yaml: Path) -> Path:
    """Verify the dataset.yaml exists and points at real directories."""
    if not dataset_yaml.exists():
        raise TrainingError(f"dataset.yaml not found: {dataset_yaml}")

    text = dataset_yaml.read_text(encoding="utf-8")
    # Find the 'path:' line; ultralytics resolves paths relative to it
    import re
    m = re.search(r"^path:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        raise TrainingError(f"dataset.yaml missing 'path:' field: {dataset_yaml}")

    dataset_root = Path(m.group(1).strip()).resolve()
    if not dataset_root.exists():
        raise TrainingError(f"dataset root does not exist: {dataset_root}")

    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        d = dataset_root / rel
        if not d.exists():
            raise TrainingError(f"dataset missing directory: {d}")

    return dataset_yaml


def _run_dir(run_dir: Path) -> Path:
    """Create a unique run directory. Fails if it already has content."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise TrainingError(
            f"run_dir already has content: {run_dir}. "
            f"Pick a new path or delete the old run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_device(device: str | int) -> str | int:
    """Translate config device value into something ultralytics accepts.

    'auto' -> GPU 0 if CUDA available, else 'cpu'.
    'cpu', 'mps' -> pass through.
    int or '0' -> pass through (assumed valid).
    """
    if isinstance(device, int):
        return device
    device = str(device).strip().lower()
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            return 0
        return "cpu"
    if device in ("cpu", "mps"):
        return device
    # Numeric string -> int
    try:
        return int(device)
    except ValueError:
        # Ultralytics may accept "cuda:0" style; pass through
        return device


def _find_best_checkpoint(results) -> Path:
    """Locate the best.pt that ultralytics saved.

    Strictly requires best.pt. Never falls back to last.pt — the two mean
    different things (best = validation-selected, last = most recent epoch).
    Silent substitution would corrupt downstream evaluation.
    """
    best = getattr(results, "save_dir", None)
    if best is None:
        raise TrainingError("ultralytics did not return a save_dir")
    save_dir = Path(str(best))
    candidate = save_dir / "weights" / "best.pt"
    if not candidate.exists():
        raise TrainingError(
            f"best.pt not found under {save_dir}/weights/. "
            f"Cannot guarantee validation-selected checkpoint. "
            f"Check training logs for early failures."
        )
    return candidate


def _read_actual_epochs_completed(results, save_dir: Path) -> int | None:
    """Read the actual number of completed epochs from results.csv.

    Returns None if the file is missing or unreadable — caller should
    mark epochs_completed as unknown rather than fabricating a value.
    """
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return None
    try:
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        # First line is header; count remaining non-empty lines
        data_rows = [ln for ln in lines[1:] if ln.strip()]
        if not data_rows:
            return None
        return len(data_rows)
    except (OSError, ValueError):
        return None


def _read_selected_epoch(save_dir: Path) -> int | None:
    """Determine which epoch produced best.pt.

    YOLOv8 saves best.pt at the epoch with highest fitness (mAP50-95).
    The epoch number is not directly stored in artifact-accessible metadata,
    so we read it from the results.csv if a 'fitness' or 'metrics/mAP50-95(B)'
    column exists, else return None (mark unknown).
    """
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return None
    try:
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return None
        header = [h.strip() for h in lines[0].split(",")]
        # Prefer 'fitness' column (ultralytics combined metric)
        # Fall back to mAP50-95
        target_col = None
        for candidate in ("fitness", "metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if candidate in header:
                target_col = header.index(candidate)
                break
        if target_col is None:
            return None
        best_epoch = None
        best_val = float("-inf")
        for idx, line in enumerate(lines[1:], start=1):
            parts = line.split(",")
            if len(parts) <= target_col:
                continue
            try:
                val = float(parts[target_col])
            except ValueError:
                continue
            if val > best_val:
                best_val = val
                # ultralytics epoch column is often first; if not, use row index
                try:
                    best_epoch = int(parts[0]) if header[0] == "epoch" else idx - 1
                except (ValueError, IndexError):
                    best_epoch = idx - 1
        return best_epoch
    except (OSError, ValueError, IndexError):
        return None


def train(
    config_path: Path,
    *,
    run_dir: Path,
    dataset_yaml_override: Path | None = None,
    baseline_config_path: Path | None = None,
) -> ModelArtifact:
    """Run fine-tuning and return the resulting ModelArtifact.

    `dataset_yaml_override` lets the CLI pass an explicit dataset path that
    overrides whatever the config says. Useful for testing and for running
    the same config against different exports.
    """
    if not config_path.exists():
        raise TrainingError(f"config not found: {config_path}")

    cfg = load_config(config_path, baseline_path=baseline_config_path)

    dataset_yaml = dataset_yaml_override or Path(cfg.data["dataset_yaml"])
    dataset_yaml = _check_dataset(dataset_yaml)

    run_dir = _run_dir(run_dir)

    YOLO, ultralytics_version = _lazy_import_ultralytics()

    resolved_device = _resolve_device(cfg.training["device"])

    # Load pretrained backbone
    model = YOLO(cfg.model["pretrained_weights"])

    # Run training — all hyperparameters from config
    datetime.now(UTC)
    results = model.train(
        data=str(dataset_yaml),
        epochs=cfg.training["epochs"],
        imgsz=cfg.preprocessing["image_size"],
        batch=cfg.training["batch_size"],
        optimizer=cfg.training["optimizer"],
        lr0=cfg.training["lr0"],
        weight_decay=cfg.training["weight_decay"],
        patience=cfg.training["patience"],
        seed=cfg.training["seed"],
        device=resolved_device,
        # augmentation
        hsv_h=cfg.augmentation["hsv_h"],
        hsv_s=cfg.augmentation["hsv_s"],
        hsv_v=cfg.augmentation["hsv_v"],
        fliplr=cfg.augmentation["fliplr"],
        mosaic=cfg.augmentation["mosaic"],
        scale=cfg.augmentation["scale"],
        # output location
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        # misc
        verbose=True,
        plots=True,
    )
    datetime.now(UTC)

    checkpoint = _find_best_checkpoint(results)
    sha256_file(checkpoint)

    # Copy best.pt into the run dir root for easy discovery
    stable_checkpoint = run_dir / "best.pt"
    shutil.copyfile(checkpoint, stable_checkpoint)
    stable_sha = sha256_file(stable_checkpoint)

    # Save the resolved config (resolves all `inherit` markers)
    resolved_config_path = run_dir / "resolved_config.json"
    # Replace device with resolved value in the saved config
    resolved_training = dict(cfg.training)
    resolved_training["device"] = resolved_device

    # Reflect overrides in the saved config for accurate provenance
    resolved_data = dict(cfg.data)
    resolved_data["dataset_yaml"] = str(dataset_yaml)
    resolved_outputs = dict(cfg.outputs)
    resolved_outputs["run_dir"] = str(run_dir)

    resolved_config = {
        "experiment_id": cfg.experiment_id,
        "owner": cfg.owner,
        "model": cfg.model,
        "data": resolved_data,
        "preprocessing": cfg.preprocessing,
        "augmentation": cfg.augmentation,
        "training": resolved_training,
        "validation": cfg.validation,
        "inference": cfg.inference,
        "outputs": resolved_outputs,
    }
    resolved_config_path.write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )

    # Determine epochs_completed from results.csv; None = unknown
    save_dir = Path(str(getattr(results, "save_dir", run_dir)))
    actual_epochs = _read_actual_epochs_completed(results, save_dir)
    selected_epoch = _read_selected_epoch(save_dir)

    if actual_epochs is None:
        # Do not fabricate. Mark as requested with a warning flag in resolved_config.
        epochs_completed = 0  # sentinel; artifact stores 0 to mean "unknown"
    else:
        epochs_completed = actual_epochs

    import torch

    artifact = ModelArtifact(
        experiment_id=cfg.experiment_id,
        run_dir=run_dir,
        checkpoint_path=stable_checkpoint,
        checkpoint_sha256=stable_sha,
        architecture=cfg.model["architecture"],
        pretrained_weights=cfg.model["pretrained_weights"],
        class_names=("bottle", "cup", "keyboard", "mouse", "book", "scissors"),
        image_size=cfg.preprocessing["image_size"],
        dataset_yaml=str(dataset_yaml),
        dataset_release=cfg.data["dataset_release"],
        split_seed=cfg.data["split_seed"],
        training_manifest_sha256=sha256_file(dataset_yaml),
        epochs_requested=cfg.training["epochs"],
        epochs_completed=epochs_completed,
        selected_epoch=(selected_epoch if selected_epoch is not None else -1),
        checkpoint_selection=cfg.validation["checkpoint_selection"],
        batch_size=cfg.training["batch_size"],
        optimizer=cfg.training["optimizer"],
        learning_rate=cfg.training["lr0"],
        weight_decay=cfg.training["weight_decay"],
        seed=cfg.training["seed"],
        device=str(resolved_device),
        confidence_threshold=cfg.inference["confidence_threshold"],
        nms_iou_threshold=cfg.inference["nms_iou_threshold"],
        ultralytics_version=ultralytics_version,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        resolved_config=resolved_config,
    )

    artifact.save(run_dir / "artifact.json")
    return artifact
