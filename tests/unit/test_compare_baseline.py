"""Tests for scripts/compare_baseline.py using synthetic fakes (no weights, no models)."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "compare_baseline_cli", SCRIPTS / "compare_baseline.py"
)
assert _SPEC is not None and _SPEC.loader is not None
compare = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare  # dataclasses resolve their module at import time
_SPEC.loader.exec_module(compare)

BOX = [0.0, 0.0, 10.0, 10.0]
PROJECT_NAMES = ("bottle", "cup", "keyboard", "mouse", "book", "scissors")
DEFAULT_POSITIONS = {name: index + 1 for index, name in enumerate(PROJECT_NAMES)}


def _names(positions: dict[str, int], total: int = 80) -> dict[int, str]:
    names = {index: f"filler_{index}" for index in range(total)}
    for name, index in positions.items():
        names[index] = name
    return names


def _manifest(split: str = "val") -> dict:
    ground_truth = {
        "p1": [{"pair_id": "p1", "class_id": 0, "status": "unchanged",
                "before_box": list(BOX), "after_box": list(BOX)}],
        "p2": [{"pair_id": "p2", "class_id": 1, "status": "removed",
                "before_box": list(BOX), "after_box": None}],
    }
    pairs = [
        {"pair_id": pair_id, "before_path": f"photos/{pair_id}_b.png",
         "after_path": f"photos/{pair_id}_a.png"}
        for pair_id in sorted(ground_truth)
    ]
    return {
        "schema_version": "1.0",
        "dataset_release": "v1",
        "split": split,
        "pairs": pairs,
        "ground_truth": ground_truth,
        "eligible_no_change_pair_ids": ["p1"],
        "manifest_sha256": "m" * 64,
    }


def _records(removed_correct: bool) -> dict[str, list[dict]]:
    return {
        "p1": [{"class_id": 0, "status": "unchanged",
                "before_box": list(BOX), "after_box": list(BOX)}],
        "p2": ([{"class_id": 1, "status": "removed",
                 "before_box": list(BOX), "after_box": None}]
               if removed_correct else []),
    }


class _Service:
    def __init__(self, records: dict, *, explode: bool = False) -> None:
        self._records = records
        self._explode = explode
        self.calls: list[str] = []

    def compare(self, *, before_path, after_path, pair_id: str) -> dict:
        self.calls.append(pair_id)
        if self._explode:
            raise RuntimeError("simulated inference failure")
        return {"pair_id": pair_id, "records": [dict(r) for r in self._records.get(pair_id, [])]}


def _settings(tmp_path: Path) -> SimpleNamespace:
    checkpoint = tmp_path / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-e0-weights")
    return SimpleNamespace(
        checkpoint_path=checkpoint.resolve(),
        checkpoint_sha256=hashlib.sha256(b"fake-e0-weights").hexdigest(),
        config_sha256="c" * 64,
        rules_version="rules-v1",
        confidence_threshold=0.2,
        movement_threshold=0.02152951369355096,
        nms_iou_threshold=0.35,
        imgsz=640,
        mode="real",
    )


def _write_weights(tmp_path: Path, name: str = "yolov8n.pt") -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(b"fake-baseline-weights")
    return path, hashlib.sha256(b"fake-baseline-weights").hexdigest()


def _run(
    tmp_path: Path,
    *,
    positions: dict[str, int] | None = None,
    manifest: dict | None = None,
    baseline_sha: str | None = None,
    e0_sha: str | None = None,
    baseline_records: dict | None = None,
    e0_records: dict | None = None,
    e0_explodes: bool = False,
    event_iou_threshold: float = 0.5,
    read_backend=None,
    out_name: str = "out",
) -> tuple[dict, dict]:
    settings = _settings(tmp_path)
    baseline, baseline_digest = _write_weights(tmp_path)
    built: dict = {}

    def build(*, weights, settings, info, mapping):
        role = "baseline" if mapping is not None else "e0"
        built[role] = {"weights": Path(weights), "mapping": mapping, "info": info,
                       "settings": settings}
        if role == "e0" and e0_explodes:
            return _Service({}, explode=True)
        return _Service(
            (baseline_records or _records(removed_correct=False)) if role == "baseline"
            else (e0_records or _records(removed_correct=True))
        )

    names = _names(positions or DEFAULT_POSITIONS)
    result = compare.run_comparison(
        baseline_weights=baseline,
        baseline_sha256=baseline_sha or baseline_digest,
        e0_weights=settings.checkpoint_path,
        e0_sha256=e0_sha or settings.checkpoint_sha256,
        config_path=tmp_path / "app.json",
        manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / out_name,
        event_iou_threshold=event_iou_threshold,
        manifest_loader=lambda _path: manifest or _manifest(),
        settings_loader=lambda _path: settings,
        read_backend=read_backend or (
            lambda _path: compare.BackendInfo(names=names, version="9.9.9")
        ),
        build_service=build,
    )
    return result, built


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_writes_mapping_table_and_comparison(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    output = tmp_path / "out"

    assert result["split"] == "val" and result["pair_count"] == 2
    assert sorted(path.name for path in output.iterdir()) == [
        "baseline", "baseline_class_mapping.csv", "baseline_class_mapping.json",
        "comparison.csv", "comparison.json", "e0",
    ]
    mapping = _json(output / "baseline_class_mapping.json")
    assert mapping["weights_sha256"] == hashlib.sha256(b"fake-baseline-weights").hexdigest()
    assert mapping["weights_path"].endswith("yolov8n.pt")
    assert len(mapping["names"]) == 80
    assert mapping["mapping"] == {str(index + 1): index for index in range(6)}
    assert len(mapping["dropped_classes"]) == 74

    rows = _csv(output / "baseline_class_mapping.csv")
    assert len(rows) == 80
    assert sum(1 for row in rows if row["status"] == "mapped") == 6
    assert sum(1 for row in rows if row["status"] == "dropped") == 74


def test_comparison_records_required_metadata_and_both_models(tmp_path: Path) -> None:
    _, built = _run(tmp_path)
    payload = _json(tmp_path / "out" / "comparison.json")

    assert payload["split"] == "val"
    assert payload["dataset_release"] == "v1"
    assert payload["manifest_sha256"] == "m" * 64
    assert payload["event_iou_threshold"] == 0.5
    assert payload["config"]["confidence_threshold"] == 0.2
    assert payload["config"]["nms_iou_threshold"] == 0.35
    assert payload["config"]["movement_threshold"] == 0.02152951369355096
    assert payload["config"]["imgsz"] == 640
    assert payload["models"]["baseline"]["ultralytics_version"] == "9.9.9"
    assert len(payload["models"]["baseline"]["checkpoint_sha256"]) == 64
    assert len(payload["models"]["e0"]["checkpoint_sha256"]) == 64
    assert "NOT held-out" in payload["selection_bias"]
    assert built["baseline"]["settings"] is built["e0"]["settings"]


def test_metrics_are_reported_per_model(tmp_path: Path) -> None:
    _run(tmp_path)
    payload = _json(tmp_path / "out" / "comparison.json")
    baseline = payload["metrics"]["baseline"]
    e0 = payload["metrics"]["e0"]

    assert baseline["macro_f1"] == pytest.approx(0.25)
    assert baseline["change_only_macro_f1"] == pytest.approx(0.0)
    # Macro F1 averages the fixed four-status set, so absent statuses score zero.
    assert e0["macro_f1"] == pytest.approx(0.5)
    assert e0["change_only_macro_f1"] == pytest.approx(1 / 3)
    assert e0["no_change"]["eligible_pair_count"] == 1
    assert e0["no_change"]["false_alarm_pair_count"] == 0

    rows = {row["metric"]: row for row in _csv(tmp_path / "out" / "comparison.csv")}
    assert rows["macro_f1"]["baseline"] == "0.250000"
    assert rows["macro_f1"]["e0"] == "0.500000"


def test_baseline_service_receives_the_derived_mapping(tmp_path: Path) -> None:
    _, built = _run(tmp_path, positions={"bottle": 10, "cup": 20, "keyboard": 30,
                                         "mouse": 40, "book": 50, "scissors": 60})
    mapping = built["baseline"]["mapping"]

    assert dict(mapping.class_map) == {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}
    assert built["e0"]["mapping"] is None


def test_mapping_follows_permuted_names(tmp_path: Path) -> None:
    permuted = {"bottle": 60, "cup": 50, "keyboard": 40, "mouse": 30,
                "book": 20, "scissors": 10}
    _, built = _run(tmp_path, positions=permuted)

    assert dict(built["baseline"]["mapping"].class_map) == {
        60: 0, 50: 1, 40: 2, 30: 3, 20: 4, 10: 5,
    }
    rows = {row["baseline_name"]: row for row in _csv(
        tmp_path / "out" / "baseline_class_mapping.csv"
    )}
    assert rows["bottle"]["baseline_index"] == "60"


def test_refuses_baseline_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline weights SHA-256 mismatch"):
        _run(tmp_path, baseline_sha="0" * 64)
    assert not (tmp_path / "out").exists()


def test_refuses_e0_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="e0 weights SHA-256 mismatch"):
        _run(tmp_path, e0_sha="0" * 64)
    assert not (tmp_path / "out").exists()


def test_refuses_non_val_manifest_before_touching_weights(tmp_path: Path) -> None:
    def forbidden(_path):
        raise AssertionError("weights must not be read for a non-val manifest")

    with pytest.raises(ValueError, match="refusing non-val manifest"):
        _run(tmp_path, manifest=_manifest(split="test"), read_backend=forbidden)
    assert not (tmp_path / "out").exists()


def test_refuses_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _run(tmp_path)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


def test_failed_run_leaves_no_partial_output(tmp_path: Path) -> None:
    # The runner wraps per-pair failures; the point is that nothing is published.
    with pytest.raises(RuntimeError, match="Evaluation failed for pair 'p1'"):
        _run(tmp_path, e0_explodes=True)

    assert not (tmp_path / "out").exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".out-")) == []


@pytest.mark.parametrize("threshold", [0, 1.5, -0.1])
def test_rejects_invalid_event_iou_threshold(tmp_path: Path, threshold: float) -> None:
    with pytest.raises(ValueError, match="event_iou_threshold"):
        _run(tmp_path, event_iou_threshold=threshold)
    assert not (tmp_path / "out").exists()


def test_missing_weights_are_reported(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="baseline weights not found"):
        compare.run_comparison(
            baseline_weights=tmp_path / "absent.pt",
            baseline_sha256="0" * 64,
            e0_weights=settings.checkpoint_path,
            e0_sha256=settings.checkpoint_sha256,
            config_path=tmp_path / "app.json",
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "out",
            event_iou_threshold=0.5,
            manifest_loader=lambda _path: _manifest(),
            settings_loader=lambda _path: settings,
        )


def _real_settings(tmp_path: Path):
    """Build a genuinely validated RealSettings over a synthetic checkpoint."""
    from deskcheck.application.real_settings import RealSettings

    checkpoint = tmp_path / "e0" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-e0-weights")
    config = {
        "schema_version": "1.0",
        "mode": "real",
        "checkpoint_path": "e0/best.pt",
        "checkpoint_sha256": hashlib.sha256(b"fake-e0-weights").hexdigest(),
        "model_origin": "synthetic fixture",
        "confidence_threshold": 0.20,
        "movement_threshold": 0.02152951369355096,
        "nms_iou_threshold": 0.35,
        "imgsz": 640,
        "rules_version": "rules-v1",
        "class_names": list(PROJECT_NAMES),
    }
    path = tmp_path / "app.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return RealSettings.load(path)


def test_real_factory_wires_baseline_detector_without_loading_a_model(
    tmp_path: Path,
) -> None:
    from deskcheck.application.comparison_service import ComparisonService
    from deskcheck.evaluation.baseline_mapping import map_baseline_names

    settings = _real_settings(tmp_path)
    weights, _ = _write_weights(tmp_path)
    info = compare.BackendInfo(
        names=_names({"bottle": 10, "cup": 20, "keyboard": 30, "mouse": 40,
                      "book": 50, "scissors": 60}),
        version="9.9.9",
    )

    service = compare.build_comparison_service(
        weights=weights, settings=settings, info=info,
        mapping=map_baseline_names(info.names),
    )

    assert isinstance(service, ComparisonService)
    assert service.config is settings


def test_real_factory_rejects_e0_weights_that_disagree_with_the_config(
    tmp_path: Path,
) -> None:
    settings = _real_settings(tmp_path)
    other, _ = _write_weights(tmp_path, name="other.pt")
    info = compare.BackendInfo(names=_names(DEFAULT_POSITIONS), version="9.9.9")

    with pytest.raises(ValueError, match="disagree with the configured checkpoint"):
        compare.build_comparison_service(
            weights=other, settings=settings, info=info, mapping=None
        )
