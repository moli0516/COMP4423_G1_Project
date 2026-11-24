"""C must detect a silently changed batch or a different training recipe."""

import importlib.util
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).parents[2] / "scripts" / "score_c_resolution_pair.py"
    assert path.is_file(), "No checkpoint-backed matched-run comparison"
    spec = importlib.util.spec_from_file_location("c_pair_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(size):
    return {
        "imgsz": size,
        "epochs": 100,
        "batch": 2,
        "seed": 42,
        "device": 0,
        "model": "yolov8n.pt",
        "data": "dataset.yaml",
        "lr0": 0.01,
        "project": "runs",
        "name": f"arm_{size}",
        "save_dir": f"runs/arm_{size}",
    }


def test_same_training_recipe_across_sizes():
    load_script().check_recipes(settings(640), settings(960), 2, 2)


@pytest.mark.parametrize(
    "field,value",
    [("batch", 1), ("lr0", 0.005), ("device", "cpu"), ("data", "other.yaml"), ("imgsz", 640)],
)
def test_mismatched_backend_settings_raise(field, value):
    candidate = {**settings(960), field: value}
    with pytest.raises(ValueError, match=field):
        load_script().check_recipes(settings(640), candidate, 2, 2)


def test_same_automatic_batch_downgrade_is_reported():
    with pytest.raises(ValueError, match="batch"):
        load_script().check_recipes(settings(640), settings(960), 8, 8)


def valid_evidence(size):
    strings = {"torch": "2.11.0", "ultralytics": "8.4.215", "python": "3.12.10"}
    receipt = {
        "best_sha": str(size)[0] * 64,
        "init_sha": "b" * 64,
        "effective_train_val_sha256": "c" * 64,
        "data_yaml_sha": "d" * 64,
        "source_trainer_sha": "e" * 64,
        "size": size,
        "batch": 2,
        "epochs_requested": 100,
        "device": "NVIDIA GeForce RTX 2060 SUPER",
        "gpu_uuid": "GPU-example",
        "cuda": "12.8",
        "actual_amp": True,
        "actual_workers": 0,
        "torch_steps": 35,
        "exit_code": 0,
        **strings,
    }
    artifact = {
        "checkpoint_sha256": receipt["best_sha"],
        "epochs_requested": 100,
        "batch_size": 2,
        "training_manifest_sha256": receipt["data_yaml_sha"],
        "torch_version": strings["torch"],
        "ultralytics_version": strings["ultralytics"],
        "python_version": strings["python"],
        "image_size": size,
        "device": "0",
    }
    return receipt, artifact


@pytest.mark.parametrize(
    "field,value",
    [
        ("init_sha", "2" * 64),
        ("effective_train_val_sha256", "3" * 64),
        ("gpu_uuid", "GPU-different"),
        ("source_trainer_sha", "4" * 64),
        ("actual_amp", False),
        ("actual_workers", 1),
        ("best_sha", "5" * 64),
        ("exit_code", 1),
        ("torch_steps", 0),
        ("torch_steps", None),
    ],
)
def test_receipts_bind_checkpoints_and_match_actual_training(field, value):
    receipts, artifacts = map(list, zip(valid_evidence(640), valid_evidence(960), strict=True))
    receipts[1][field] = value
    with pytest.raises(ValueError, match=field):
        load_script().check_provenance(receipts, artifacts)


def test_cpu_uninstrumented_steps_require_machine_identity_and_explicit_method():
    receipts, artifacts = map(list, zip(valid_evidence(640), valid_evidence(960), strict=True))
    for evidence, artifact in zip(receipts, artifacts, strict=True):
        evidence.update(
            device="cpu",
            gpu_uuid=None,
            cuda=None,
            actual_amp=False,
            torch_steps=None,
            step_count_method="not instrumented",
            machine_id="linux-A",
        )
        artifact["device"] = "cpu"
    scorer = load_script()
    result = scorer.check_provenance(receipts, artifacts)
    assert result["machine_id"] == "linux-A" and result["torch_steps"] == [None, None]
    receipts[0]["machine_id"] = "linux-B"
    with pytest.raises(ValueError, match="machine_id"):
        scorer.check_provenance(receipts, artifacts)
    receipts[0]["machine_id"] = "linux-A"
    del receipts[0]["step_count_method"]
    with pytest.raises(ValueError, match="torch_steps"):
        scorer.check_provenance(receipts, artifacts)


def test_receipt_lookup_for_flat_and_nested_outputs(tmp_path):
    runner = tmp_path / "arm"
    run = runner / "output"
    run.mkdir(parents=True)
    script = load_script()
    with pytest.raises(FileNotFoundError):
        script.read_receipt(run, None)
    (runner / "receipt.json").write_text('{"scope": "wrapper"}')
    assert script.read_receipt(run, None) == {"scope": "wrapper"}
    (run / "receipt.json").write_text('{"scope": "flat"}')
    assert script.read_receipt(run, None) == {"scope": "flat"}
    assert script.read_receipt(run, runner / "receipt.json") == {"scope": "wrapper"}
