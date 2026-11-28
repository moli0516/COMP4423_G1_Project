from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from deskcheck.training import e1_experiment as e1
from deskcheck.training.e1_experiment import (
    APPROVAL_FIELDS,
    APPROVAL_TOKEN,
    build_receipt,
    execute_e1,
    hash_initialization,
    implementation_identity,
    inventory_dataset,
    resolve_plan,
    sha256_file,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _synthetic_runtime_metadata(monkeypatch):
    environment = e1._environment_identity()
    environment["packages"].update({"ultralytics": "synthetic", "torch": "synthetic"})
    monkeypatch.setattr(e1, "_environment_identity", lambda: environment)


def _configs(tmp_path: Path, baseline_extra: dict | None = None, treatment_extra: dict | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / "configs/experiments/baseline.json").read_text())
    if baseline_extra:
        baseline["augmentation"].update(baseline_extra)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    treatment = json.loads((ROOT / "configs/experiments/augmentation.json").read_text())
    if treatment_extra:
        treatment["augmentation"].update(treatment_extra)
    treatment_path = tmp_path / "treatment.json"
    treatment_path.write_text(json.dumps(treatment), encoding="utf-8")
    return baseline_path, treatment_path


def _pair(pair_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "pair_id": pair_id,
        "capture_group_id": f"group-{pair_id}",
        "before_image_id": f"{pair_id}-before",
        "after_image_id": f"{pair_id}-after",
        "before_image_path": f"raw/{pair_id}-before.jpg",
        "after_image_path": f"raw/{pair_id}-after.jpg",
        "width": 640,
        "height": 480,
        "rules_version": "1.0",
        "independently_captured_no_change": False,
        "manual_before_objects": [],
        "manual_after_objects": [],
        "correspondences": [],
        "events": [],
    }


def _dataset(tmp_path: Path, *, include_test_key: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset_root / f"images/{split}").mkdir(parents=True)
        (dataset_root / f"labels/{split}").mkdir(parents=True)
    pairs = [_pair(f"{name}_pair") for name in ("train", "val", "test")]
    for split, pair in zip(("train", "val", "test"), pairs, strict=True):
        for side in ("before", "after"):
            stem = f"{pair['pair_id']}__{side}"
            (dataset_root / f"images/{split}/{stem}.jpg").write_bytes(f"{split}-{side}".encode())
            (dataset_root / f"labels/{split}/{stem}.txt").write_text("", encoding="utf-8")

    annotations = tmp_path / "pairs.jsonl"
    annotations.write_text(
        "".join(json.dumps(pair) + "\n" for pair in pairs),
        encoding="utf-8",
    )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({
        "schema_version": "1.0",
        "seed": 42,
        "dataset_release": "v1",
        "splits": {
            "train": ["train_pair"],
            "val": ["val_pair"],
            "test": ["test_pair"],
        },
    }), encoding="utf-8")
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test_key:
        lines.append("test: images/test")
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_yaml, annotations, splits, dataset_root


def _plan(tmp_path: Path):
    baseline, treatment = _configs(tmp_path)
    return resolve_plan(baseline, treatment, ROOT, baseline), baseline, treatment


def _approved():
    return {name: True for name in APPROVAL_FIELDS}


def _build_complete(
    tmp_path: Path,
    *,
    approval_records: dict[str, bool] | None = None,
    run_relative: str = "new-run",
):
    plan, baseline, treatment = _plan(tmp_path)
    dataset_yaml, annotations, splits, _ = _dataset(tmp_path)
    initialization = tmp_path / "initialization.pt"
    initialization.write_bytes(b"synthetic initialization")
    receipt_dir = tmp_path / "receipt"
    run_dir = tmp_path / run_relative
    receipt = build_receipt(
        ROOT, plan, initialization, dataset_yaml, receipt_dir,
        annotations_path=annotations,
        split_manifest_path=splits,
        run_dir=run_dir,
        approval_records=approval_records if approval_records is not None else _approved(),
    )
    receipt_path = write_receipt(receipt, receipt_dir)
    return receipt, receipt_path.parent / "execution_specification.json", initialization, dataset_yaml, annotations, splits, run_dir, baseline, treatment


def _rewrite_linked_specification(spec_path: Path, mutate_spec, mutate_receipt=None):
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    mutate_spec(specification)
    spec_path.write_text(json.dumps(specification), encoding="utf-8")
    receipt_path = spec_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution_specification"] = specification
    receipt["execution_specification_sha256"] = e1.canonical_json_sha256(specification)
    receipt["intended_output_locations"]["execution"] = specification.get("run_dir")
    if mutate_receipt is not None:
        mutate_receipt(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def test_inheritance_and_exact_six_zero(tmp_path):
    baseline, treatment = _configs(tmp_path)
    plan = resolve_plan(baseline, treatment, ROOT, baseline)
    assert plan["six_zero_policy_valid"] is True
    assert plan["allowed_differences"]
    assert plan["forbidden_differences"] == {}
    assert plan["treatment"]["training"] == plan["control"]["training"]


def test_forbidden_non_scientific_change(tmp_path):
    baseline, treatment = _configs(tmp_path)
    changed = json.loads(treatment.read_text())
    changed["training"] = {"epochs": 50}
    treatment.write_text(json.dumps(changed), encoding="utf-8")
    plan = resolve_plan(baseline, treatment, ROOT, baseline)
    assert "training.epochs" in plan["forbidden_differences"]


def test_noop_augmentation_key(tmp_path):
    baseline, treatment = _configs(tmp_path)
    base = json.loads(baseline.read_text())
    base["augmentation"]["hsv_h"] = 0.0
    baseline.write_text(json.dumps(base), encoding="utf-8")
    plan = resolve_plan(baseline, treatment, ROOT, baseline)
    assert "augmentation.hsv_h" in plan["no_op_augmentation_keys"]


def test_identical_and_differing_extra_augmentation(tmp_path):
    baseline, treatment = _configs(tmp_path, {"degrees": 10.0}, {"degrees": 10.0})
    plan = resolve_plan(baseline, treatment, ROOT, baseline)
    assert "augmentation.degrees" not in plan["forbidden_differences"]
    baseline, treatment = _configs(tmp_path, {"degrees": 10.0}, {"degrees": 15.0})
    plan = resolve_plan(baseline, treatment, ROOT, baseline)
    assert "augmentation.degrees" in plan["forbidden_differences"]


def test_initialization_hash(tmp_path):
    init = tmp_path / "weights.pt"
    init.write_bytes(b"weights")
    first = hash_initialization(init)
    assert len(first["sha256"]) == 64
    init.write_bytes(b"different")
    assert hash_initialization(init)["sha256"] != first["sha256"]


def test_inventory_requires_catalog_and_maps_disjoint_splits(tmp_path):
    dataset_yaml, annotations, splits, _ = _dataset(tmp_path)
    inventory = inventory_dataset(dataset_yaml, ROOT, annotations, splits)
    assert inventory["status"] == "ready"
    assert {row["split"] for row in inventory["members"]} == {"train", "val"}
    assert inventory["split_catalog"]["split_membership_sha256"]
    assert len(inventory["members"]) == 4


def test_inventory_rejects_train_val_manifest_overlap(tmp_path):
    dataset_yaml, annotations, splits, _ = _dataset(tmp_path)
    manifest = json.loads(splits.read_text(encoding="utf-8"))
    manifest["splits"]["val"] = ["train_pair"]
    splits.write_text(json.dumps(manifest), encoding="utf-8")
    inventory = inventory_dataset(dataset_yaml, ROOT, annotations, splits)
    assert inventory["status"] == "incomplete"
    assert "appears in both" in inventory["error"]


def test_inventory_rejects_missing_val_label_and_unpaired_label(tmp_path):
    dataset_yaml, annotations, splits, root = _dataset(tmp_path)
    (root / "labels/val/val_pair__before.txt").unlink()
    assert inventory_dataset(dataset_yaml, ROOT, annotations, splits)["status"] == "incomplete"

    dataset_yaml, annotations, splits, root = _dataset(tmp_path / "extra")
    (root / "labels/train/unpaired.txt").write_text("", encoding="utf-8")
    assert inventory_dataset(dataset_yaml, ROOT, annotations, splits)["status"] == "incomplete"


@pytest.mark.parametrize("target_split", ["train", "val"])
def test_symlink_to_test_is_rejected_before_hashing_test_content(tmp_path, monkeypatch, target_split):
    dataset_yaml, annotations, splits, root = _dataset(tmp_path)
    image_dir = root / f"images/{target_split}"
    label_dir = root / f"labels/{target_split}"
    shutil.rmtree(image_dir)
    shutil.rmtree(label_dir)
    image_dir.symlink_to(root / "images/test", target_is_directory=True)
    label_dir.symlink_to(root / "labels/test", target_is_directory=True)
    dataset_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )

    hashed: list[Path] = []
    real_hash = e1.sha256_file

    def observe_hash(path: Path) -> str:
        hashed.append(path.resolve())
        return real_hash(path)

    monkeypatch.setattr(e1, "sha256_file", observe_hash)
    inventory = inventory_dataset(dataset_yaml, ROOT, annotations, splits)
    assert inventory["status"] == "incomplete"
    assert not any(path.is_relative_to(root / "images/test") for path in hashed)
    assert not any(path.is_relative_to(root / "labels/test") for path in hashed)


def test_train_val_aliasing_and_wrong_split_directories_rejected(tmp_path):
    dataset_yaml, annotations, splits, root = _dataset(tmp_path)
    shutil.rmtree(root / "images/val")
    shutil.rmtree(root / "labels/val")
    (root / "images/val").symlink_to(root / "images/train", target_is_directory=True)
    (root / "labels/val").symlink_to(root / "labels/train", target_is_directory=True)
    inventory = inventory_dataset(dataset_yaml, ROOT, annotations, splits)
    assert inventory["status"] == "incomplete"

    dataset_yaml, annotations, splits, root = _dataset(tmp_path / "swapped")
    saved_images = tmp_path / "saved_images"
    saved_labels = tmp_path / "saved_labels"
    shutil.copytree(root / "images/train", saved_images / "train", dirs_exist_ok=True)
    shutil.copytree(root / "images/val", saved_images / "val", dirs_exist_ok=True)
    shutil.copytree(root / "labels/train", saved_labels / "train", dirs_exist_ok=True)
    shutil.copytree(root / "labels/val", saved_labels / "val", dirs_exist_ok=True)
    for split in ("train", "val"):
        shutil.rmtree(root / f"images/{split}")
        shutil.rmtree(root / f"labels/{split}")
    for target, source in (("train", "val"), ("val", "train")):
        shutil.copytree(saved_images / source, root / f"images/{target}")
        shutil.copytree(saved_labels / source, root / f"labels/{target}")
    inventory = inventory_dataset(dataset_yaml, ROOT, annotations, splits)
    assert inventory["status"] == "incomplete"
    assert "assigned to" in inventory["error"]


def test_unsupported_layout_rejected_and_shared_trainer_accepts_supported_layout(
    monkeypatch, tmp_path
):
    from deskcheck.training import trainer as trainer_module
    from deskcheck.training.trainer import TrainingError, _check_dataset, train

    dataset_yaml, annotations, splits, root = _dataset(tmp_path)
    _check_dataset(dataset_yaml)
    assert inventory_dataset(dataset_yaml, ROOT, annotations, splits)["status"] == "ready"

    dataset_yaml.write_text(f"path: {root}\n", encoding="utf-8")
    assert inventory_dataset(dataset_yaml, ROOT, annotations, splits)["status"] == "incomplete"
    with pytest.raises(TrainingError, match="explicit train"):
        _check_dataset(dataset_yaml)

    dataset_yaml.write_text(
        f"path: {root}\ntrain: custom/train\nval: custom/val\n",
        encoding="utf-8",
    )
    inventory = inventory_dataset(dataset_yaml, ROOT, annotations, splits)
    assert inventory["status"] == "incomplete"
    assert "train target must be images/train" in inventory["error"]

    with pytest.raises(TrainingError, match="train target must be"):
        _check_dataset(dataset_yaml)

    baseline, treatment = _configs(tmp_path)
    backend_calls = []
    monkeypatch.setattr(
        trainer_module,
        "_lazy_import_ultralytics",
        lambda: backend_calls.append("imported"),
    )
    run_dir = tmp_path / "unsupported-layout-run"
    with pytest.raises(TrainingError, match="train target must be"):
        train(
            treatment,
            run_dir=run_dir,
            dataset_yaml_override=dataset_yaml,
            baseline_config_path=baseline,
        )
    assert backend_calls == []
    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("layout", "error"),
    [
        ("{path}\nval: images/val\n", "explicit train"),
        ("{path}\ntrain: images/train\n", "explicit val"),
        ("{path}\n", "explicit train"),
        ("{path}\ntrain: custom/train\nval: images/val\n", "train target"),
        ("{path}\ntrain: images/train\nval: custom/val\n", "val target"),
        ("{path}\ntrain: custom/train\nval: custom/val\n", "train target"),
        ("{path}\ntrain: images/train\nval: images/val\ntrain_labels: labels/custom\n", "label targets"),
        ("{path}\ntrain: images/train\nval: images/val\nval_labels: labels/custom\n", "label targets"),
    ],
)
def test_dataset_layout_failures_are_consistent_and_pre_backend(
    monkeypatch, tmp_path, layout, error
):
    from deskcheck.training import trainer as trainer_module
    from deskcheck.training.trainer import TrainingError, _check_dataset, train

    dataset_yaml, annotations, splits, root = _dataset(tmp_path)
    dataset_yaml.write_text(layout.format(path=f"path: {root}"), encoding="utf-8")
    inventory = inventory_dataset(dataset_yaml, ROOT, annotations, splits)
    assert inventory["status"] == "incomplete"
    assert error in inventory["error"]
    with pytest.raises(TrainingError, match=error):
        _check_dataset(dataset_yaml)

    baseline, treatment = _configs(tmp_path)
    backend_calls = []
    monkeypatch.setattr(
        trainer_module,
        "_lazy_import_ultralytics",
        lambda: backend_calls.append("imported"),
    )
    run_dir = tmp_path / "must-not-be-created"
    with pytest.raises(TrainingError, match=error):
        train(
            treatment,
            run_dir=run_dir,
            dataset_yaml_override=dataset_yaml,
            baseline_config_path=baseline,
        )
    assert backend_calls == []
    assert not run_dir.exists()


def test_missing_metadata_is_incomplete(tmp_path):
    dataset_yaml, _, _, _ = _dataset(tmp_path)
    assert inventory_dataset(dataset_yaml, ROOT)["status"] == "incomplete"


def test_readiness_truth_table_and_spec_identity(tmp_path):
    receipt, spec_path, _, _, _, _, _, _, _ = _build_complete(
        tmp_path, approval_records={name: False for name in APPROVAL_FIELDS}
    )
    assert receipt["technical_ready"] is True
    assert receipt["formal_training_approved"] is False
    assert receipt["execution_intent"] is False
    assert receipt["execution_permission"] is False
    assert receipt["readiness"] == "approval_incomplete"
    spec = json.loads(spec_path.read_text())
    assert receipt["execution_specification_sha256"] == e1.canonical_json_sha256(spec)
    assert spec["execution_permission"] is False


def test_complete_preflight_saves_explicit_approval_but_not_execution_intent(tmp_path):
    receipt, _, _, _, _, _, _, _, _ = _build_complete(tmp_path)
    assert receipt["technical_ready"] is True
    assert receipt["formal_training_approved"] is True
    assert receipt["execution_intent"] is False
    assert receipt["execution_permission"] is False
    assert receipt["readiness"] == "ready_for_explicit_execution_intent"
    assert "training_performed" in receipt and receipt["training_performed"] is False


def test_execution_uses_frozen_inputs_and_forwards_same_initialization(tmp_path):
    _, spec_path, initialization, _, _, _, run_dir, _, treatment = _build_complete(tmp_path)
    assert not run_dir.exists()
    calls = {}

    def trainer(config_path, **kwargs):
        calls["config_path"] = config_path
        calls.update(kwargs)
        return "stub-result"

    assert execute_e1(spec_path, approval_token=APPROVAL_TOKEN, trainer=trainer) == "stub-result"
    assert calls["config_path"] == treatment.resolve()
    assert calls["initialization_override"] == initialization.resolve()
    assert calls["expected_initialization_sha256"] == sha256_file(initialization)
    assert calls["run_dir"] == run_dir.resolve()
    assert not run_dir.exists()


def test_preflight_initialization_a_rejects_changed_bytes_and_relocation(tmp_path):
    _, spec_path, initialization, _, _, _, _, _, _ = _build_complete(tmp_path)
    other = tmp_path / "initialization-B.pt"
    other.write_bytes(b"initialization B")
    trainer_calls = []
    with pytest.raises(TypeError):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            initialization_override=other,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []
    with pytest.raises(TypeError):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            run_dir=tmp_path / "unplanned-run",
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []
    initialization.write_bytes(other.read_bytes())
    with pytest.raises(RuntimeError, match="frozen specification"):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []


def test_missing_initialization_cannot_authorize_execution(tmp_path):
    plan, _, _ = _plan(tmp_path)
    dataset_yaml, annotations, splits, _ = _dataset(tmp_path)
    receipt_dir = tmp_path / "receipt"
    receipt = build_receipt(
        ROOT, plan, None, dataset_yaml, receipt_dir,
        annotations_path=annotations,
        split_manifest_path=splits,
        run_dir=tmp_path / "new-run",
        approval_records=_approved(),
    )
    assert receipt["technical_ready"] is False
    assert receipt["formal_training_approved"] is False
    assert "initialization bytes/identity missing" in receipt["missing_prerequisites"]
    receipt_path = write_receipt(receipt, receipt_dir)
    trainer_calls = []
    with pytest.raises(RuntimeError, match="not technically ready"):
        execute_e1(
            receipt_path.parent / "execution_specification.json",
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []


def test_missing_backend_runtime_blocks_technical_readiness(monkeypatch, tmp_path):
    environment = e1._environment_identity()
    environment["packages"].update({"ultralytics": None, "torch": None})
    monkeypatch.setattr(e1, "_environment_identity", lambda: environment)
    plan, _, _ = _plan(tmp_path)
    dataset_yaml, annotations, splits, _ = _dataset(tmp_path)
    initialization = tmp_path / "init.pt"
    initialization.write_bytes(b"init")
    receipt = build_receipt(
        ROOT, plan, initialization, dataset_yaml, tmp_path / "receipt",
        annotations_path=annotations,
        split_manifest_path=splits,
        run_dir=tmp_path / "new-run",
        approval_records=_approved(),
    )
    assert receipt["technical_ready"] is False
    assert receipt["formal_training_approved"] is False
    assert "required runtime package missing: ultralytics" in receipt["missing_prerequisites"]


def test_stale_config_and_specification_are_rejected(tmp_path):
    _, spec_path, _, _, _, _, _, _, treatment = _build_complete(tmp_path)
    original = spec_path.read_text()
    treatment.write_text(treatment.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="config bytes changed"):
        execute_e1(spec_path, approval_token=APPROVAL_TOKEN, trainer=lambda *args, **kwargs: None)
    specification = json.loads(original)
    specification["test_tampering_marker"] = "changed"
    spec_path.write_text(json.dumps(specification), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        execute_e1(spec_path, approval_token=APPROVAL_TOKEN, trainer=lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("receipt_version", "3", "Unsupported preflight receipt_version"),
        ("receipt_version", 2, "Unsupported preflight receipt_version"),
        ("receipt_version", True, "Unsupported preflight receipt_version"),
        ("receipt_version", None, "missing required fields"),
    ],
)
def test_execution_rejects_malformed_or_unsupported_receipt_version(
    tmp_path, field, value, error
):
    _, spec_path, *_ = _build_complete(tmp_path)
    receipt_path = spec_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if value is None:
        receipt.pop(field)
    else:
        receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    trainer_calls = []
    with pytest.raises(RuntimeError, match=error):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda spec: spec.update(schema_version="1"), "Unsupported execution specification schema_version"),
        (lambda spec: spec.update(schema_version=2), "Unsupported execution specification schema_version"),
        (lambda spec: spec.update(schema_version=True), "Unsupported execution specification schema_version"),
        (lambda spec: spec.pop("run_dir"), "missing required fields"),
        (lambda spec: spec.update(run_dir=7), "execution specification.run_dir must have type"),
        (lambda spec: spec.update(technical_ready="true"), "execution specification.technical_ready must have type"),
        (lambda spec: spec["approval_records"].update(owner_scientific_approval="yes"), "approval_records.owner_scientific_approval must have type"),
    ],
)
def test_execution_rejects_malformed_specification_schema(tmp_path, change, error):
    _, spec_path, *_ = _build_complete(tmp_path)
    _rewrite_linked_specification(spec_path, change)
    trainer_calls = []
    with pytest.raises(RuntimeError, match=error):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []


@pytest.mark.parametrize(
    "embedded_change",
    [
        lambda embedded: embedded.update(run_dir="/contradictory/receipt-copy"),
        lambda embedded: embedded["source_identity"].update({"synthetic": "different"}),
    ],
)
def test_execution_rejects_receipt_embedded_spec_disagreement(
    tmp_path, embedded_change
):
    _, spec_path, *_ = _build_complete(tmp_path)
    separate_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    receipt_path = spec_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    embedded_change(receipt["execution_specification"])
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert json.loads(spec_path.read_text(encoding="utf-8")) == separate_spec
    assert receipt["execution_specification_sha256"] == e1.canonical_json_sha256(separate_spec)
    trainer_calls = []
    with pytest.raises(RuntimeError, match="embedded execution specification differs"):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []


def test_execution_rejects_missing_receipt_field_and_incorrect_field_type(tmp_path):
    _, spec_path, *_ = _build_complete(tmp_path)
    receipt_path = spec_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    del receipt["execution_permission"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="preflight receipt is missing required fields"):
        execute_e1(spec_path, approval_token=APPROVAL_TOKEN, trainer=lambda *args, **kwargs: None)

    _, spec_path, *_ = _build_complete(tmp_path / "wrong-type")
    receipt_path = spec_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution_permission"] = 0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="preflight receipt.execution_permission must have type"):
        execute_e1(spec_path, approval_token=APPROVAL_TOKEN, trainer=lambda *args, **kwargs: None)


def test_execution_rejects_changed_dataset_catalog(tmp_path):
    _, spec_path, _, _, _, splits, _, _, _ = _build_complete(tmp_path)
    manifest = json.loads(splits.read_text(encoding="utf-8"))
    manifest["seed"] = 99
    splits.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset or authoritative split membership"):
        execute_e1(spec_path, approval_token=APPROVAL_TOKEN, trainer=lambda *args, **kwargs: None)


@pytest.mark.parametrize("missing_approval", APPROVAL_FIELDS)
def test_execution_requires_each_approval_record(tmp_path, missing_approval):
    records = _approved()
    records[missing_approval] = False
    _, spec_path, _, _, _, _, _, _, _ = _build_complete(tmp_path, approval_records=records)
    with pytest.raises(RuntimeError, match="approval records"):
        execute_e1(spec_path, approval_token=APPROVAL_TOKEN, trainer=lambda *args, **kwargs: None)


def test_execution_requires_explicit_token(tmp_path):
    _, spec_path, _, _, _, _, _, _, _ = _build_complete(tmp_path / "token")
    with pytest.raises(RuntimeError, match="execution-intent"):
        execute_e1(spec_path, approval_token="wrong", trainer=lambda *args, **kwargs: None)


@pytest.mark.parametrize("collision", ["file", "directory", "symlink"])
def test_execution_rejects_frozen_output_collisions(tmp_path, collision):
    receipt, spec_path, _, _, _, _, run_dir, _, _ = _build_complete(tmp_path)
    if collision == "file":
        run_dir.write_text("collision", encoding="utf-8")
    elif collision == "directory":
        run_dir.mkdir()
    else:
        target = tmp_path / "target"
        target.mkdir()
        run_dir.symlink_to(target, target_is_directory=True)
    trainer_calls = []
    with pytest.raises(RuntimeError, match="must still be nonexistent"):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []
    assert receipt["intended_output_locations"]["execution"] == receipt["execution_specification"]["run_dir"]


@pytest.mark.parametrize("destination", ["receipt", "outside"])
def test_execution_rejects_redirected_run_ancestor_before_trainer(tmp_path, destination):
    parent = tmp_path / "run-parent"
    parent.mkdir()
    _, spec_path, _, _, _, _, run_dir, *_ = _build_complete(
        tmp_path, run_relative="run-parent/new-run"
    )
    parked_parent = tmp_path / "original-run-parent"
    parent.rename(parked_parent)
    if destination == "receipt":
        redirect_target = spec_path.parent
    else:
        redirect_target = tmp_path / "unrelated-target"
        redirect_target.mkdir()
    parent.symlink_to(redirect_target, target_is_directory=True)

    trainer_calls = []
    with pytest.raises(RuntimeError, match="different path|symlink path component"):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []
    assert not run_dir.exists()
    assert not (redirect_target / "new-run").exists()


@pytest.mark.parametrize("nesting", ["run_contains_receipt", "receipt_contains_run"])
def test_execution_rechecks_receipt_run_nesting_before_trainer(tmp_path, nesting):
    _, spec_path, *_ = _build_complete(tmp_path)
    if nesting == "run_contains_receipt":
        execution_path = spec_path.parent.parent
    else:
        execution_path = spec_path.parent / "nested-run"
    _rewrite_linked_specification(
        spec_path,
        lambda specification: specification.update(
            run_dir=str(execution_path.resolve(strict=False))
        ),
    )
    trainer_calls = []
    with pytest.raises(RuntimeError, match="must not be nested"):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
        )
    assert trainer_calls == []


def test_execution_spec_cannot_be_built_with_existing_run_dir(tmp_path):
    plan, _, _ = _plan(tmp_path)
    dataset_yaml, annotations, splits, _ = _dataset(tmp_path)
    initialization = tmp_path / "init.pt"
    initialization.write_bytes(b"init")
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    receipt = build_receipt(
        ROOT, plan, initialization, dataset_yaml, tmp_path / "receipt",
        annotations_path=annotations,
        split_manifest_path=splits,
        run_dir=run_dir,
        approval_records=_approved(),
    )
    assert receipt["technical_ready"] is False
    assert receipt["execution_specification"]["run_dir"] is None


@pytest.mark.parametrize("receipt_inside_run", [True, False])
def test_execution_and_receipt_directories_must_not_nest(tmp_path, receipt_inside_run):
    plan, _, _ = _plan(tmp_path)
    dataset_yaml, annotations, splits, _ = _dataset(tmp_path)
    initialization = tmp_path / "init.pt"
    initialization.write_bytes(b"init")
    shared = tmp_path / "shared"
    if receipt_inside_run:
        run_dir = shared
        receipt_dir = shared / "receipt"
    else:
        run_dir = shared / "run"
        receipt_dir = shared

    receipt = build_receipt(
        ROOT, plan, initialization, dataset_yaml, receipt_dir,
        annotations_path=annotations,
        split_manifest_path=splits,
        run_dir=run_dir,
        approval_records=_approved(),
    )
    assert receipt["technical_ready"] is False
    assert "execution output directory must be separate from receipt output" in receipt["missing_prerequisites"]


def test_receipt_output_collision_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "receipt-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        write_receipt({"execution_specification": {}}, link)


def test_cli_approval_flags_cannot_authorize_incomplete_preflight(tmp_path):
    from scripts import preflight_e1

    receipt_dir = tmp_path / "receipt"
    result = preflight_e1.main([
        "--output", str(receipt_dir),
        "--owner-approved",
        "--six-key-policy-approved",
        "--control-comparability-approved",
        "--effective-settings-reviewed",
        "--backend-defaults-verified",
        "--downstream-rules-frozen",
    ])
    assert result == 2
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["technical_ready"] is False
    assert receipt["formal_training_approved"] is False
    assert receipt["execution_intent"] is False
    assert receipt["execution_permission"] is False


@pytest.mark.parametrize(
    "override",
    [
        ["--run-dir", "other-run"],
        ["--initialization", "other-init.pt"],
        ["--control", "other-control.json"],
        ["--output", "other-receipt"],
        ["--owner-approved"],
    ],
)
def test_execute_cli_rejects_preflight_overrides(monkeypatch, tmp_path, override):
    from scripts import preflight_e1

    calls = []
    monkeypatch.setattr(
        preflight_e1,
        "execute_e1",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = preflight_e1.main([
        "--execute",
        "--specification", str(tmp_path / "saved-specification.json"),
        "--approval-token", APPROVAL_TOKEN,
        *override,
    ])
    assert result == 1
    assert calls == []


def test_dry_run_guards_training_boundary_and_does_not_make_run_directory(monkeypatch, tmp_path):
    from deskcheck.training import trainer as trainer_module

    def forbidden(*args, **kwargs):
        raise AssertionError("backend/training boundary invoked by preflight")

    monkeypatch.setattr(trainer_module, "train", forbidden)
    monkeypatch.setattr(trainer_module, "_lazy_import_ultralytics", forbidden)
    from scripts import preflight_e1

    run_dir = tmp_path / "must-not-exist"
    receipt_dir = tmp_path / "receipt"
    result = preflight_e1.main([
        "--output", str(receipt_dir),
        "--run-dir", str(run_dir),
    ])
    assert result == 2
    assert not run_dir.exists()
    saved = json.loads((receipt_dir / "receipt.json").read_text())
    assert saved["training_performed"] is False
    assert saved["execution_permission"] is False


def test_trainer_rejects_hash_mismatch_before_backend_or_model(monkeypatch, tmp_path):
    from deskcheck.training import trainer as trainer_module
    from deskcheck.training.trainer import TrainingError, train

    baseline, treatment = _configs(tmp_path)
    dataset_yaml, _, _, _ = _dataset(tmp_path)
    init = tmp_path / "init.pt"
    init.write_bytes(b"changed")
    invoked = []

    def forbidden_backend():
        invoked.append("backend")
        raise AssertionError("backend imported")

    monkeypatch.setattr(trainer_module, "_lazy_import_ultralytics", forbidden_backend)
    with pytest.raises(TrainingError, match="bytes changed"):
        train(
            treatment,
            run_dir=tmp_path / "not-created",
            dataset_yaml_override=dataset_yaml,
            baseline_config_path=baseline,
            initialization_override=init,
            expected_initialization_sha256=sha256_file(init) + "0",
        )
    assert invoked == []
    assert not (tmp_path / "not-created").exists()


def test_trainer_rechecks_initialization_immediately_before_model_construction(monkeypatch, tmp_path):
    from deskcheck.training import trainer as trainer_module
    from deskcheck.training.trainer import TrainingError, train

    baseline, treatment = _configs(tmp_path)
    dataset_yaml, _, _, _ = _dataset(tmp_path)
    initialization = tmp_path / "init.pt"
    initialization.write_bytes(b"initial")
    expected_sha256 = sha256_file(initialization)
    invoked = []

    class Model:
        def __init__(self, path):
            invoked.append(("model", path))

    def load_backend():
        initialization.write_bytes(b"changed between boundary checks")
        invoked.append(("backend", None))
        return Model, "stub"

    monkeypatch.setattr(trainer_module, "_lazy_import_ultralytics", load_backend)
    monkeypatch.setattr(trainer_module, "_resolve_device", lambda device: device)
    with pytest.raises(TrainingError, match="before model construction"):
        train(
            treatment,
            run_dir=tmp_path / "not-created",
            dataset_yaml_override=dataset_yaml,
            baseline_config_path=baseline,
            initialization_override=initialization,
            expected_initialization_sha256=expected_sha256,
        )
    assert invoked == [("backend", None)]


def test_trainer_rejects_missing_initialization_before_backend(monkeypatch, tmp_path):
    from deskcheck.training import trainer as trainer_module
    from deskcheck.training.trainer import TrainingError, train

    baseline, treatment = _configs(tmp_path)
    invoked = []
    monkeypatch.setattr(
        trainer_module,
        "_lazy_import_ultralytics",
        lambda: invoked.append("backend"),
    )
    with pytest.raises(TrainingError, match="initialization not found"):
        train(
            treatment,
            run_dir=tmp_path / "not-created",
            baseline_config_path=baseline,
            initialization_override=tmp_path / "missing.pt",
            expected_initialization_sha256="a" * 64,
        )
    assert invoked == []


def test_shared_trainer_legacy_optional_initialization_default_is_preserved():
    import inspect

    from deskcheck.training.trainer import train

    parameters = inspect.signature(train).parameters
    assert parameters["initialization_override"].default is None
    assert parameters["expected_initialization_sha256"].default is None


def test_shared_trainer_forwards_augmentation_and_sets_runs_dir(monkeypatch, tmp_path):
    from deskcheck.training import trainer as trainer_module
    from deskcheck.training.trainer import train

    baseline = json.loads((ROOT / "configs/experiments/baseline.json").read_text())
    baseline["augmentation"].update({"degrees": 3.0, "translate": 0.1})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(baseline), encoding="utf-8")
    initialization = tmp_path / "local.pt"
    initialization.write_bytes(b"synthetic init")
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("synthetic", encoding="utf-8")

    calls = {}
    settings_values = {"runs_dir": "original"}

    class Settings:
        def update(self, values):
            settings_values.update(values)

    class StopBeforeTraining(Exception):
        pass

    class FakeModel:
        def __init__(self, path):
            calls["model_initialization"] = path

        def train(self, **kwargs):
            calls["train_kwargs"] = kwargs
            raise StopBeforeTraining

    module = ModuleType("ultralytics")
    module.settings = Settings()
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(trainer_module, "_lazy_import_ultralytics", lambda: (FakeModel, "stub"))
    monkeypatch.setattr(trainer_module, "_check_dataset", lambda path: path)
    monkeypatch.setattr(trainer_module, "_resolve_device", lambda device: device)

    with pytest.raises(StopBeforeTraining):
        train(
            config_path,
            run_dir=tmp_path / "new-run",
            dataset_yaml_override=dataset_yaml,
            initialization_override=initialization,
            expected_initialization_sha256=sha256_file(initialization),
        )

    assert calls["model_initialization"] == str(initialization.resolve())
    assert calls["train_kwargs"]["degrees"] == 3.0
    assert calls["train_kwargs"]["translate"] == 0.1
    assert calls["train_kwargs"]["hsv_h"] == baseline["augmentation"]["hsv_h"]
    assert settings_values["runs_dir"] == str((tmp_path / "new-run").parent.resolve())


def test_shared_trainer_artifact_records_bound_initialization(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from deskcheck.training import trainer as trainer_module
    from deskcheck.training.trainer import train

    baseline = json.loads((ROOT / "configs/experiments/baseline.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(baseline), encoding="utf-8")
    initialization = tmp_path / "local.pt"
    initialization.write_bytes(b"synthetic init")
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("synthetic", encoding="utf-8")
    settings_values = {}

    class Settings:
        def update(self, values):
            settings_values.update(values)

    class FakeModel:
        def __init__(self, path):
            assert path == str(initialization.resolve())

        def train(self, **kwargs):
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"fake checkpoint")
            (save_dir / "results.csv").write_text(
                "epoch,fitness\n0,0.5\n",
                encoding="utf-8",
            )
            return SimpleNamespace(save_dir=save_dir)

    ultralytics_module = ModuleType("ultralytics")
    ultralytics_module.settings = Settings()
    torch_module = ModuleType("torch")
    torch_module.__version__ = "stub"
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(trainer_module, "_lazy_import_ultralytics", lambda: (FakeModel, "stub"))
    monkeypatch.setattr(trainer_module, "_check_dataset", lambda path: path)
    monkeypatch.setattr(trainer_module, "_resolve_device", lambda device: "cpu")

    artifact = train(
        config_path,
        run_dir=tmp_path / "artifact-run",
        dataset_yaml_override=dataset_yaml,
        initialization_override=initialization,
        expected_initialization_sha256=sha256_file(initialization),
    )
    assert artifact.resolved_config["model"]["pretrained_weights"] == str(initialization.resolve())
    assert artifact.resolved_config["model"]["pretrained_weights_sha256"] == sha256_file(initialization)
    assert artifact.checkpoint_sha256 == sha256_file(artifact.checkpoint_path)
    assert artifact.selected_epoch == 0
    assert settings_values["runs_dir"] == str((tmp_path / "artifact-run").parent.resolve())


def test_receipt_persists_relevant_source_identity(tmp_path):
    receipt, spec_path, *_ = _build_complete(tmp_path)
    assert receipt["receipt_version"] == 3
    assert receipt["implementation_identity"] == implementation_identity(ROOT)
    assert "src/deskcheck/training/dataset_layout.py" in receipt["implementation_identity"]
    assert json.loads(spec_path.read_text()) == receipt["execution_specification"]
    assert len(set(receipt["implementation_identity"].values())) >= 4


def test_execution_rejects_stale_source_identity(monkeypatch, tmp_path):
    _, spec_path, *_ = _build_complete(tmp_path)
    original_identity = e1.implementation_identity
    monkeypatch.setattr(
        e1,
        "implementation_identity",
        lambda root=ROOT: {
            **original_identity(root),
            "synthetic-source.py": "changed",
        },
    )
    with pytest.raises(RuntimeError, match="source identity differs"):
        execute_e1(
            spec_path,
            approval_token=APPROVAL_TOKEN,
            trainer=lambda *args, **kwargs: None,
        )
