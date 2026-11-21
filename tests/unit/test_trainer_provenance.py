"""Regression tests for trainer checkpoint/epoch provenance (B's review)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from deskcheck.training.trainer import (
    TrainingError,
    _find_best_checkpoint,
    _read_actual_epochs_completed,
    _read_selected_epoch,
)


class _FakeResults:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir


def test_find_best_checkpoint_requires_best_pt(tmp_path: Path) -> None:
    """Missing best.pt must raise, not silently fall back to last.pt."""
    save_dir = tmp_path / "run"
    weights = save_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"fake last")

    results = _FakeResults(save_dir)
    with pytest.raises(TrainingError, match="best.pt not found"):
        _find_best_checkpoint(results)


def test_find_best_checkpoint_returns_best_pt(tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    weights = save_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"fake best")
    (weights / "last.pt").write_bytes(b"fake last")

    results = _FakeResults(save_dir)
    result = _find_best_checkpoint(results)
    assert result.name == "best.pt"


def test_read_actual_epochs_completed_from_results_csv(tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    csv_path = save_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train/box_loss", "metrics/mAP50-95(B)"])
        for i in range(5):
            writer.writerow([i, 1.0 - i * 0.1, 0.5 + i * 0.05])

    results = _FakeResults(save_dir)
    assert _read_actual_epochs_completed(results, save_dir) == 5


def test_read_actual_epochs_completed_missing_csv_returns_none(tmp_path: Path) -> None:
    save_dir = tmp_path / "empty"
    save_dir.mkdir()
    results = _FakeResults(save_dir)
    assert _read_actual_epochs_completed(results, save_dir) is None


def test_read_selected_epoch_picks_max_fitness(tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    csv_path = save_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "fitness"])
        writer.writerow([0, 0.3])
        writer.writerow([1, 0.5])
        writer.writerow([2, 0.7])  # best
        writer.writerow([3, 0.6])
        writer.writerow([4, 0.55])

    assert _read_selected_epoch(save_dir) == 2


def test_read_selected_epoch_missing_csv_returns_none(tmp_path: Path) -> None:
    save_dir = tmp_path / "empty"
    save_dir.mkdir()
    assert _read_selected_epoch(save_dir) is None


def test_read_selected_epoch_missing_epoch_column_returns_none(tmp_path: Path) -> None:
    """No 'epoch' column -> unknown. Must NOT fall back to row index."""
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    csv_path = save_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "fitness"])  # no 'epoch' column
        writer.writerow([0, 0.3])
        writer.writerow([1, 0.5])
    assert _read_selected_epoch(save_dir) is None


def test_read_selected_epoch_non_integer_epoch_returns_none(tmp_path: Path) -> None:
    """Non-integer epoch values -> unknown. Must NOT fall back to row index."""
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    csv_path = save_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "fitness"])
        writer.writerow(["abc", 0.3])   # non-integer epoch
        writer.writerow([1, 0.5])
    assert _read_selected_epoch(save_dir) is None


def test_read_selected_epoch_valid_returns_actual_epoch(tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    csv_path = save_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "fitness"])
        writer.writerow([5, 0.5])
        writer.writerow([6, 0.9])   # best
        writer.writerow([7, 0.7])
    assert _read_selected_epoch(save_dir) == 6
