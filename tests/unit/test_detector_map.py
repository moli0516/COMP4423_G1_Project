"""Unit tests for detector_map.evaluate_detector_map (fake validator only)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pytest

from deskcheck.evaluation.detector_map import EXPECTED_CLASS_NAMES, evaluate_detector_map

REPORT_FIELDS = {
    "map50",
    "map50_95",
    "per_class",
    "class_names",
    "split",
    "weights_sha256",
    "dataset_yaml_sha256",
    "backend_name",
    "backend_version",
    "utc_timestamp",
}


def _write_dataset(root: Path) -> Path:
    for split in ("val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "a.jpg").write_bytes(b"original-image-bytes")
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(EXPECTED_CLASS_NAMES)}\nnames: {list(EXPECTED_CLASS_NAMES)}\n",
        encoding="utf-8",
    )
    return dataset_yaml


def _metrics(**overrides: object) -> dict:
    result = {
        "backend_name": "fake-backend",
        "backend_version": "0.0.0",
        "map50": 0.5,
        "map50_95": 0.25,
        "class_names": list(EXPECTED_CLASS_NAMES),
        "per_class": {
            name: {"ap50": 0.5, "ap50_95": 0.25} for name in EXPECTED_CLASS_NAMES
        },
    }
    result.update(overrides)
    return result


def _validator(result: object, calls: list) -> object:
    def validate(*, weights: Path, dataset_yaml: Path, split: str) -> object:
        calls.append({"weights": weights, "dataset_yaml": dataset_yaml, "split": split})
        return result

    return validate


def _kwargs(tmp_path: Path, dataset_yaml: Path, validator: object, **overrides: object) -> dict:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    values = {
        "weights": weights,
        "dataset_yaml": dataset_yaml,
        "split": "val",
        "output": tmp_path / "reports" / "map_val.json",
        "copy_dir": tmp_path / "dataset_copy",
        "validator": validator,
    }
    values.update(overrides)
    return values


def _yaml_path_value(dataset_yaml: Path) -> str:
    text = dataset_yaml.read_text(encoding="utf-8")
    match = re.search(r"^path\s*:\s*(.*)$", text, flags=re.MULTILINE)
    assert match is not None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def test_success_writes_report_values(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    calls: list = []
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(), calls))

    report = evaluate_detector_map(**kwargs)

    assert report["map50"] == 0.5
    assert report["map50_95"] == 0.25
    assert report["per_class"]["bottle"] == {"ap50": 0.5, "ap50_95": 0.25}
    assert report["class_names"] == list(EXPECTED_CLASS_NAMES)
    assert report["split"] == "val"
    assert report["backend_name"] == "fake-backend"
    assert report["backend_version"] == "0.0.0"
    assert report["weights_sha256"] == hashlib.sha256(b"fake-weights").hexdigest()
    assert report["dataset_yaml_sha256"] == hashlib.sha256(dataset_yaml.read_bytes()).hexdigest()
    assert datetime.fromisoformat(report["utc_timestamp"]).utcoffset() is not None
    assert len(calls) == 1
    assert calls[0]["split"] == "val"


def test_report_json_has_every_required_field(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(), []))

    report = evaluate_detector_map(**kwargs)

    written = json.loads(Path(kwargs["output"]).read_text(encoding="utf-8"))
    assert set(written) == REPORT_FIELDS
    assert written == report
    assert set(written["per_class"]) == set(EXPECTED_CLASS_NAMES)
    assert all(set(entry) == {"ap50", "ap50_95"} for entry in written["per_class"].values())
    assert all(isinstance(written[key], float) for key in ("map50", "map50_95"))


def test_missing_metric_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    broken = _metrics()
    del broken["map50"]
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(broken, []))

    with pytest.raises(ValueError, match="map50"):
        evaluate_detector_map(**kwargs)
    assert not Path(kwargs["output"]).exists()


def test_nan_metric_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(map50=math.nan), []))

    with pytest.raises(ValueError, match="finite"):
        evaluate_detector_map(**kwargs)
    assert not Path(kwargs["output"]).exists()


def test_infinite_per_class_metric_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    per_class = {name: {"ap50": 0.5, "ap50_95": 0.25} for name in EXPECTED_CLASS_NAMES}
    per_class["cup"]["ap50_95"] = math.inf
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(per_class=per_class), []))

    with pytest.raises(ValueError, match="finite"):
        evaluate_detector_map(**kwargs)


def test_wrong_class_list_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    kwargs = _kwargs(
        tmp_path,
        dataset_yaml,
        _validator(_metrics(class_names=["bottle", "cup", "keyboard", "mouse", "book", "person"]), []),
    )

    with pytest.raises(ValueError, match="approved detector classes"):
        evaluate_detector_map(**kwargs)
    assert not Path(kwargs["output"]).exists()


def test_missing_per_class_entry_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    per_class = {name: {"ap50": 0.5, "ap50_95": 0.25} for name in EXPECTED_CLASS_NAMES}
    del per_class["scissors"]
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(per_class=per_class), []))

    with pytest.raises(ValueError, match="missing class"):
        evaluate_detector_map(**kwargs)


def test_existing_output_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "reports" / "map_val.json"
    output.parent.mkdir()
    output.write_text("{}", encoding="utf-8")
    calls: list = []
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(), calls), output=output)

    with pytest.raises(FileExistsError):
        evaluate_detector_map(**kwargs)
    assert calls == []
    assert not Path(kwargs["copy_dir"]).exists()


def test_existing_copy_dir_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    copy_dir = tmp_path / "dataset_copy"
    copy_dir.mkdir()
    calls: list = []
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(), calls), copy_dir=copy_dir)

    with pytest.raises(FileExistsError):
        evaluate_detector_map(**kwargs)
    assert calls == []
    assert not Path(kwargs["output"]).exists()


def test_test_split_without_confirmation_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    calls: list = []
    kwargs = _kwargs(
        tmp_path, dataset_yaml, _validator(_metrics(), calls), split="test"
    )

    with pytest.raises(ValueError, match="confirm_final_test"):
        evaluate_detector_map(**kwargs)
    assert calls == []
    assert not Path(kwargs["output"]).exists()
    assert not Path(kwargs["copy_dir"]).exists()


def test_test_split_with_confirmation_runs_on_test(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    calls: list = []
    kwargs = _kwargs(
        tmp_path,
        dataset_yaml,
        _validator(_metrics(), calls),
        split="test",
        confirm_final_test=True,
    )

    report = evaluate_detector_map(**kwargs)

    assert report["split"] == "test"
    assert calls[0]["split"] == "test"


def test_validator_only_sees_the_dataset_copy(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    calls: list = []
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(), calls))

    evaluate_detector_map(**kwargs)

    copy_dir = Path(kwargs["copy_dir"]).resolve()
    copied_yaml = calls[0]["dataset_yaml"]
    assert copied_yaml == copy_dir / "dataset.yaml"
    assert copy_dir in copied_yaml.parents
    assert Path(_yaml_path_value(copied_yaml)).resolve() == copy_dir
    assert (copy_dir / "images" / "val" / "a.jpg").read_bytes() == b"original-image-bytes"


def test_original_dataset_is_unchanged_after_run(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    original_yaml = dataset_yaml.read_bytes()
    original_image = (dataset_yaml.parent / "images" / "val" / "a.jpg").read_bytes()

    def mutate_copy(*, weights: Path, dataset_yaml: Path, split: str) -> object:
        dataset_yaml.write_text("path: /backend-rewrote-this\n", encoding="utf-8")
        (dataset_yaml.parent / "images" / "val" / "a.jpg").write_bytes(b"rewritten")
        (dataset_yaml.parent / "images" / "val" / "backend_extra.jpg").write_bytes(b"extra")
        return _metrics()

    kwargs = _kwargs(tmp_path, dataset_yaml, mutate_copy)

    evaluate_detector_map(**kwargs)

    assert dataset_yaml.read_bytes() == original_yaml
    assert (dataset_yaml.parent / "images" / "val" / "a.jpg").read_bytes() == original_image
    assert not (dataset_yaml.parent / "images" / "val" / "backend_extra.jpg").exists()
    assert (Path(kwargs["copy_dir"]) / "images" / "val" / "backend_extra.jpg").exists()


def test_copy_dir_inside_dataset_root_is_rejected(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    kwargs = _kwargs(
        tmp_path,
        dataset_yaml,
        _validator(_metrics(), []),
        copy_dir=dataset_yaml.parent / "copy",
    )

    with pytest.raises(ValueError, match="outside the dataset root"):
        evaluate_detector_map(**kwargs)


def test_nonfinite_json_is_never_written(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    kwargs = _kwargs(tmp_path, dataset_yaml, _validator(_metrics(map50_95=math.nan), []))

    with pytest.raises(ValueError):
        evaluate_detector_map(**kwargs)
    assert not Path(kwargs["output"]).exists()


CLI_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_detector_map.py"


def _load_cli() -> object:
    spec = importlib.util.spec_from_file_location("evaluate_detector_map_cli", CLI_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_ultralytics(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    """Replace ultralytics with a stub: no real backend is imported or run."""

    class FakeBox:
        ap_class_index: ClassVar[list[int]] = [5, 0, 1, 2, 3, 4]
        ap50: ClassVar[list[float]] = [0.10, 0.90, 0.80, 0.70, 0.60, 0.50]
        ap: ClassVar[list[float]] = [0.05, 0.45, 0.40, 0.35, 0.30, 0.25]

    class FakeMetrics:
        names: ClassVar[dict[int, str]] = dict(enumerate(EXPECTED_CLASS_NAMES))
        box: ClassVar[FakeBox] = FakeBox()
        results_dict: ClassVar[dict[str, float]] = {
            "metrics/mAP50(B)": 0.85,
            "metrics/mAP50-95(B)": 0.50,
        }

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def val(self, *, data: str, split: str) -> FakeMetrics:
            calls.append({"weights": self.weights, "data": data, "split": split})
            return FakeMetrics()

    module = types.ModuleType("ultralytics")
    module.__version__ = "9.9.9"
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_cli_maps_fake_backend_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    output = tmp_path / "reports" / "map_val.json"
    copy_dir = tmp_path / "dataset_copy"
    backend_calls: list = []
    _fake_ultralytics(monkeypatch, backend_calls)
    cli = _load_cli()

    code = cli.main([
        "--weights", str(weights),
        "--dataset-yaml", str(dataset_yaml),
        "--split", "val",
        "--output", str(output),
        "--copy-dir", str(copy_dir),
    ])

    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["backend_name"] == "ultralytics"
    assert report["backend_version"] == "9.9.9"
    assert report["map50"] == 0.85
    assert report["map50_95"] == 0.50
    assert report["per_class"]["bottle"] == {"ap50": 0.90, "ap50_95": 0.45}
    assert report["per_class"]["scissors"] == {"ap50": 0.10, "ap50_95": 0.05}
    assert backend_calls == [
        {"weights": str(weights), "data": str(copy_dir / "dataset.yaml"), "split": "val"}
    ]


def test_cli_warns_before_held_out_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    _fake_ultralytics(monkeypatch, [])
    cli = _load_cli()

    code = cli.main([
        "--weights", str(weights),
        "--dataset-yaml", str(dataset_yaml),
        "--split", "test",
        "--output", str(tmp_path / "map_test.json"),
        "--copy-dir", str(tmp_path / "dataset_copy"),
    ])

    captured = capsys.readouterr()
    assert code == 2
    assert "Run it once" in captured.err
    assert not (tmp_path / "map_test.json").exists()
