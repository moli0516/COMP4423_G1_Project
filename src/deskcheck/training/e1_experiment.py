"""Bounded, backend-free infrastructure for the proposed DeskCheck E1 ablation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deskcheck.data.dataset_catalog import load_catalog
from deskcheck.training.dataset_layout import parse_supported_dataset_layout
from deskcheck.training.training_config import load as load_config

SIX_KEYS = ("hsv_h", "hsv_s", "hsv_v", "fliplr", "mosaic", "scale")
APPROVAL_TOKEN = "APPROVED_E1_TRAINING"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
APPROVAL_FIELDS = (
    "owner_scientific_approval",
    "six_key_policy_approval",
    "control_comparability_approval",
    "effective_settings_review",
    "backend_defaults_verified",
    "downstream_rules_frozen",
)
CODE_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_VERSION = 3
SPECIFICATION_SCHEMA_VERSION = 1
RECEIPT_REQUIRED_FIELDS = {
    "receipt_version", "training_performed", "experiment_id", "technical_ready",
    "owner_approval", "execution_intent", "execution_permission",
    "formal_training_approved", "source", "implementation_identity",
    "source_file_hashes", "plan", "execution_specification",
    "execution_specification_sha256", "initialization", "effective_train_val",
    "environment", "intended_output_locations", "missing_prerequisites",
    "readiness", "receipt_dir",
}
SPECIFICATION_REQUIRED_FIELDS = {
    "schema_version", "plan", "initialization", "dataset_yaml_path",
    "annotations_path", "split_manifest_path", "effective_train_val",
    "run_dir", "source_identity", "environment", "approval_records",
    "technical_ready", "formal_training_approved", "missing_prerequisites",
    "execution_intent", "execution_permission",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_fields(value: dict[str, Any], required: set[str], what: str) -> None:
    missing = required - value.keys()
    if missing:
        raise RuntimeError(f"Saved {what} is missing required fields: {sorted(missing)}")


def _require_type(value: Any, expected: type | tuple[type, ...], what: str) -> None:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if type(value) not in expected_types:
        names = ", ".join(item.__name__ for item in expected_types)
        raise RuntimeError(f"Saved {what} must have type {names}")


def _require_string_mapping(value: Any, what: str) -> None:
    _require_type(value, dict, what)
    if any(type(key) is not str or type(item) is not str for key, item in value.items()):
        raise RuntimeError(f"Saved {what} must map strings to strings")


def _validate_environment(value: Any, what: str) -> None:
    _require_type(value, dict, what)
    _require_fields(value, {"python", "executable", "platform", "packages"}, what)
    for key in ("python", "executable", "platform"):
        _require_type(value[key], str, f"{what}.{key}")
    _require_type(value["packages"], dict, f"{what}.packages")
    if any(
        type(key) is not str or type(package) not in (str, type(None))
        for key, package in value["packages"].items()
    ):
        raise RuntimeError(f"Saved {what}.packages must map strings to strings or null")
    if not {"ultralytics", "torch", "numpy"}.issubset(value["packages"]):
        raise RuntimeError(f"Saved {what}.packages is missing required package metadata")


def _validate_plan(value: Any, what: str) -> None:
    _require_type(value, dict, what)
    _require_fields(
        value,
        {
            "control_path", "treatment_path", "baseline_path", "config_identity",
            "control", "treatment", "allowed_differences", "forbidden_differences",
            "no_op_augmentation_keys", "six_zero_policy_valid",
        },
        what,
    )
    for key in ("control_path", "treatment_path", "baseline_path"):
        _require_type(value[key], str, f"{what}.{key}")
    _require_type(value["config_identity"], dict, f"{what}.config_identity")
    _require_fields(
        value["config_identity"],
        {"control_sha256", "treatment_sha256", "baseline_sha256"},
        f"{what}.config_identity",
    )
    _require_string_mapping(value["config_identity"], f"{what}.config_identity")
    for key in ("control", "treatment", "allowed_differences", "forbidden_differences"):
        _require_type(value[key], dict, f"{what}.{key}")
    _require_type(value["no_op_augmentation_keys"], list, f"{what}.no_op_augmentation_keys")
    if any(type(item) is not str for item in value["no_op_augmentation_keys"]):
        raise RuntimeError(f"Saved {what}.no_op_augmentation_keys must contain strings")
    _require_type(value["six_zero_policy_valid"], bool, f"{what}.six_zero_policy_valid")


def _validate_specification(value: Any) -> None:
    _require_type(value, dict, "execution specification")
    _require_fields(value, SPECIFICATION_REQUIRED_FIELDS, "execution specification")
    if type(value["schema_version"]) is not int or value["schema_version"] != SPECIFICATION_SCHEMA_VERSION:
        raise RuntimeError("Unsupported execution specification schema_version")
    _validate_plan(value["plan"], "execution specification.plan")

    initialization = value["initialization"]
    _require_type(initialization, dict, "execution specification.initialization")
    _require_fields(initialization, {"status", "path", "sha256"}, "execution specification.initialization")
    _require_type(initialization["status"], str, "execution specification.initialization.status")
    if initialization["status"] not in ("missing", "present"):
        raise RuntimeError("Saved execution specification.initialization.status is invalid")
    for key in ("path", "sha256"):
        _require_type(
            initialization[key], (str, type(None)),
            f"execution specification.initialization.{key}",
        )
    if initialization["status"] == "present" and (
        initialization["path"] is None or initialization["sha256"] is None
    ):
        raise RuntimeError("Present execution specification.initialization requires path and sha256")

    for key in ("dataset_yaml_path", "annotations_path", "split_manifest_path", "run_dir"):
        _require_type(value[key], (str, type(None)), f"execution specification.{key}")
    inventory = value["effective_train_val"]
    _require_type(inventory, dict, "execution specification.effective_train_val")
    _require_fields(inventory, {"status", "members"}, "execution specification.effective_train_val")
    _require_type(inventory["status"], str, "execution specification.effective_train_val.status")
    if inventory["status"] not in ("missing", "incomplete", "ready"):
        raise RuntimeError("Saved execution specification.effective_train_val.status is invalid")
    _require_type(inventory["members"], list, "execution specification.effective_train_val.members")
    if inventory["status"] == "ready":
        _require_fields(
            inventory,
            {"yaml_sha256", "train_val_sha256", "split_catalog", "dataset_yaml",
             "portable_root", "annotations_path", "split_manifest_path",
             "test_split_rejected"},
            "ready execution specification.effective_train_val",
        )
        for key in (
            "yaml_sha256", "train_val_sha256", "dataset_yaml", "portable_root",
            "annotations_path", "split_manifest_path",
        ):
            _require_type(
                inventory[key], str,
                f"execution specification.effective_train_val.{key}",
            )
        _require_type(inventory["test_split_rejected"], bool, "execution specification.effective_train_val.test_split_rejected")
        _require_type(
            inventory["split_catalog"], dict,
            "execution specification.effective_train_val.split_catalog",
        )
        _require_fields(
            inventory["split_catalog"],
            {"annotations_sha256", "split_manifest_sha256", "split_membership_sha256"},
            "execution specification.effective_train_val.split_catalog",
        )
        _require_string_mapping(
            inventory["split_catalog"],
            "execution specification.effective_train_val.split_catalog",
        )
        for member in inventory["members"]:
            _require_type(member, dict, "execution specification.effective_train_val member")
            _require_fields(
                member, {"member", "label", "sha256", "label_sha256", "split"},
                "execution specification.effective_train_val member",
            )
            for key in ("member", "label", "sha256", "label_sha256", "split"):
                _require_type(
                    member[key], str,
                    f"execution specification.effective_train_val member.{key}",
                )
            if member["split"] not in ("train", "val"):
                raise RuntimeError("Saved effective train/val inventory contains a non-train/val member")

    _require_string_mapping(value["source_identity"], "execution specification.source_identity")
    _validate_environment(value["environment"], "execution specification.environment")
    approvals = value["approval_records"]
    _require_type(approvals, dict, "execution specification.approval_records")
    if set(approvals) != set(APPROVAL_FIELDS):
        raise RuntimeError("Saved execution specification.approval_records has an invalid field set")
    for key in APPROVAL_FIELDS:
        _require_type(approvals[key], bool, f"execution specification.approval_records.{key}")
    for key in ("technical_ready", "formal_training_approved", "execution_intent", "execution_permission"):
        _require_type(value[key], bool, f"execution specification.{key}")
    expected_approval = value["technical_ready"] and all(approvals.values())
    if value["formal_training_approved"] is not expected_approval:
        raise RuntimeError("Saved formal_training_approved conflicts with readiness and approval records")
    if value["execution_intent"] or value["execution_permission"]:
        raise RuntimeError("Saved preflight specification cannot record execution intent or permission")
    _require_type(value["missing_prerequisites"], list, "execution specification.missing_prerequisites")
    if any(type(item) is not str for item in value["missing_prerequisites"]):
        raise RuntimeError("Saved execution specification.missing_prerequisites must contain strings")


def _validate_receipt(value: Any) -> None:
    _require_type(value, dict, "preflight receipt")
    _require_fields(value, RECEIPT_REQUIRED_FIELDS, "preflight receipt")
    if type(value["receipt_version"]) is not int or value["receipt_version"] != RECEIPT_VERSION:
        raise RuntimeError("Unsupported preflight receipt_version")
    for key in (
        "training_performed", "technical_ready", "owner_approval",
        "execution_intent", "execution_permission", "formal_training_approved",
    ):
        _require_type(value[key], bool, f"preflight receipt.{key}")
    _require_type(value["experiment_id"], str, "preflight receipt.experiment_id")
    _require_type(value["source"], dict, "preflight receipt.source")
    _require_fields(value["source"], {"sha", "branch", "dirty", "dirty_scope"}, "preflight receipt.source")
    for key in ("sha", "branch"):
        _require_type(value["source"][key], (str, type(None)), f"preflight receipt.source.{key}")
    _require_type(value["source"]["dirty"], bool, "preflight receipt.source.dirty")
    _require_type(value["source"]["dirty_scope"], str, "preflight receipt.source.dirty_scope")
    _require_string_mapping(value["implementation_identity"], "preflight receipt.implementation_identity")
    _require_string_mapping(value["source_file_hashes"], "preflight receipt.source_file_hashes")
    _validate_plan(value["plan"], "preflight receipt.plan")
    _validate_specification(value["execution_specification"])
    _require_type(value["execution_specification_sha256"], str, "preflight receipt.execution_specification_sha256")
    _require_type(value["initialization"], dict, "preflight receipt.initialization")
    _require_type(value["effective_train_val"], dict, "preflight receipt.effective_train_val")
    _validate_environment(value["environment"], "preflight receipt.environment")
    _require_type(value["intended_output_locations"], dict, "preflight receipt.intended_output_locations")
    _require_fields(
        value["intended_output_locations"], {"control", "treatment", "execution"},
        "preflight receipt.intended_output_locations",
    )
    for key in ("control", "treatment"):
        _require_type(value["intended_output_locations"][key], str, f"preflight receipt.intended_output_locations.{key}")
    _require_type(value["intended_output_locations"]["execution"], (str, type(None)), "preflight receipt.intended_output_locations.execution")
    _require_type(value["missing_prerequisites"], list, "preflight receipt.missing_prerequisites")
    if any(type(item) is not str for item in value["missing_prerequisites"]):
        raise RuntimeError("Saved preflight receipt.missing_prerequisites must contain strings")
    _require_type(value["readiness"], str, "preflight receipt.readiness")
    _require_type(value["receipt_dir"], str, "preflight receipt.receipt_dir")


def _contains_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _validate_execution_directory(run_dir_value: str, receipt_dir: Path) -> Path:
    frozen_path = Path(run_dir_value)
    if frozen_path.is_symlink():
        raise RuntimeError(f"Frozen execution directory must still be nonexistent: {frozen_path}")
    if not frozen_path.is_absolute():
        raise RuntimeError("Frozen execution directory is not an absolute canonical path")

    current_path = frozen_path.resolve(strict=False)
    if current_path != frozen_path:
        raise RuntimeError("Frozen execution directory now resolves to a different path")
    if _contains_symlink_component(frozen_path):
        raise RuntimeError("Frozen execution directory has a symlink path component")

    current_receipt_dir = receipt_dir.resolve(strict=False)
    if current_path == current_receipt_dir or current_path in current_receipt_dir.parents or current_receipt_dir in current_path.parents:
        raise RuntimeError("Execution directory and receipt directory must not be nested")
    if frozen_path.is_symlink() or frozen_path.exists():
        raise RuntimeError(f"Frozen execution directory must still be nonexistent: {frozen_path}")
    if frozen_path.resolve(strict=False) != current_path:
        raise RuntimeError("Frozen execution directory changed during final validation")
    return frozen_path


def _load_json_object(path: Path, what: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Saved {what} is unreadable or malformed: {exc}") from exc
    _require_type(value, dict, f"saved {what}")
    return value


def hash_initialization(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "missing", "path": None, "sha256": None}
    path = path.resolve()
    if not path.is_file():
        return {"status": "missing", "path": str(path), "sha256": None}
    return {"status": "present", "path": str(path), "sha256": sha256_file(path)}


def _resolve_within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes dataset root: {resolved}") from exc
    return resolved


def _catalog_membership(annotations_path: Path, split_manifest_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    pairs, manifest = load_catalog(annotations_path, split_manifest_path)
    membership: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    for split_name, pair_ids in manifest.splits.items():
        exported_image_ids: set[str] = set()
        for pair_id in pair_ids:
            pair = pairs[pair_id]
            for side, image_id in (
                ("before", pair.before_image_id),
                ("after", pair.after_image_id),
            ):
                previous_split = image_splits.get(image_id)
                if previous_split is not None and previous_split != split_name:
                    raise ValueError(
                        f"image_id {image_id!r} is assigned to both "
                        f"{previous_split!r} and {split_name!r}"
                    )
                image_splits[image_id] = split_name
                if image_id in exported_image_ids:
                    continue
                exported_image_ids.add(image_id)
                stem = f"{pair_id}__{side}"
                previous = membership.get(stem)
                if previous is not None and previous != split_name:
                    raise ValueError(f"ambiguous exported image stem: {stem}")
                membership[stem] = split_name
    identity = {
        "annotations_sha256": sha256_file(annotations_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_membership_sha256": canonical_json_sha256(membership),
    }
    return membership, identity


def _inventory_split(
    dataset_root: Path,
    split: str,
    target: Path,
    labels_target: Path,
    membership: dict[str, str],
) -> list[dict[str, Any]]:
    if not target.is_dir() or not labels_target.is_dir():
        raise ValueError(f"Missing {split} image/label directory")
    known_test_images = (dataset_root / "images/test").resolve()
    known_test_labels = (dataset_root / "labels/test").resolve()
    if target == known_test_images or target.is_relative_to(known_test_images):
        raise ValueError(f"{split} image target resolves into the known test location")
    if labels_target == known_test_labels or labels_target.is_relative_to(known_test_labels):
        raise ValueError(f"{split} label target resolves into the known test location")
    images = sorted(
        (path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(target).as_posix(),
    )
    if not images:
        raise ValueError(f"Missing {split} images")

    # Resolve all membership before opening or hashing any image or label bytes.
    assignments: list[tuple[Path, Path]] = []
    seen_stems: set[str] = set()
    for image in images:
        if image.parent.resolve() != target:
            raise ValueError(f"Nested image paths are unsupported in {split}: {image}")
        resolved_image = image.resolve()
        if resolved_image == known_test_images or resolved_image.is_relative_to(known_test_images):
            raise ValueError(f"{split} image aliases the known test location: {image.name}")
        stem = image.stem
        assigned_split = membership.get(stem)
        if assigned_split is None:
            raise ValueError(f"Unmapped {split} image member: {image.name}")
        if assigned_split == "test":
            raise ValueError(f"{split} target contains a test member: {image.name}")
        if assigned_split != split:
            raise ValueError(
                f"{split} target contains a member assigned to {assigned_split}: {image.name}"
            )
        if stem in seen_stems:
            raise ValueError(f"Duplicate {split} image member: {stem}")
        seen_stems.add(stem)
        label = (labels_target / image.relative_to(target)).with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"Missing label for {split} image: {image.name}")
        resolved_label = label.resolve()
        if resolved_label == known_test_labels or resolved_label.is_relative_to(known_test_labels):
            raise ValueError(f"{split} label aliases the known test location: {label.name}")
        assignments.append((resolved_image, resolved_label))

    labels = {
        path.resolve()
        for path in labels_target.rglob("*.txt")
        if path.is_file()
    }
    expected_labels = {label for _, label in assignments}
    if labels != expected_labels:
        raise ValueError(f"Unpaired or out-of-target labels in {split} split")

    rows: list[dict[str, Any]] = []
    for image, label in assignments:
        rows.append({
            "member": image.relative_to(dataset_root).as_posix(),
            "label": label.relative_to(dataset_root).as_posix(),
            "sha256": sha256_file(image),
            "label_sha256": sha256_file(label),
            "split": split,
        })
    return rows


def inventory_dataset(
    dataset_yaml: Path | None,
    root: Path | None = None,
    annotations_path: Path | None = None,
    split_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if dataset_yaml is None:
        return {"status": "missing", "yaml_sha256": None, "train_val_sha256": None,
                "members": [], "test_split_rejected": False, "split_catalog": None}
    yaml_path = (root / dataset_yaml).resolve() if root and not dataset_yaml.is_absolute() else dataset_yaml.resolve()
    if not yaml_path.is_file():
        return {"status": "missing", "yaml_sha256": None, "train_val_sha256": None,
                "members": [], "test_split_rejected": False, "dataset_yaml": str(yaml_path),
                "split_catalog": None}

    text = yaml_path.read_text(encoding="utf-8")
    try:
        layout = parse_supported_dataset_layout(text)
    except ValueError as exc:
        return {"status": "incomplete", "yaml_sha256": sha256_file(yaml_path), "train_val_sha256": None,
                "members": [], "test_split_rejected": False, "dataset_yaml": str(yaml_path),
                "split_catalog": None, "error": str(exc)}
    path_value = layout.path
    if not path_value:
        return {"status": "incomplete", "yaml_sha256": sha256_file(yaml_path), "train_val_sha256": None,
                "members": [], "test_split_rejected": False, "dataset_yaml": str(yaml_path),
                "split_catalog": None, "error": "dataset.yaml path is required by the shared trainer"}
    dataset_root = Path(path_value)
    dataset_root = dataset_root if dataset_root.is_absolute() else (yaml_path.parent / dataset_root).resolve()
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        return {"status": "incomplete", "yaml_sha256": sha256_file(yaml_path), "train_val_sha256": None,
                "members": [], "test_split_rejected": False, "dataset_yaml": str(yaml_path),
                "split_catalog": None}

    try:
        if annotations_path is None or split_manifest_path is None:
            raise ValueError("Authoritative annotations and split manifest are required")
        annotations_path = (
            root / annotations_path if root is not None and not annotations_path.is_absolute()
            else annotations_path
        ).resolve()
        split_manifest_path = (
            root / split_manifest_path if root is not None and not split_manifest_path.is_absolute()
            else split_manifest_path
        ).resolve()
        membership, catalog_identity = _catalog_membership(annotations_path, split_manifest_path)
        train_target = _resolve_within(dataset_root, dataset_root / "images/train")
        val_target = _resolve_within(dataset_root, dataset_root / "images/val")
        if train_target == val_target:
            raise ValueError("Train and val targets resolve to the same directory")
        train_labels = _resolve_within(dataset_root, dataset_root / "labels/train")
        val_labels = _resolve_within(dataset_root, dataset_root / "labels/val")
        members = _inventory_split(dataset_root, "train", train_target, train_labels, membership)
        members += _inventory_split(dataset_root, "val", val_target, val_labels, membership)
    except (OSError, ValueError) as exc:
        return {"status": "incomplete", "yaml_sha256": sha256_file(yaml_path), "train_val_sha256": None,
                "members": [], "test_split_rejected": False, "dataset_yaml": str(yaml_path),
                "split_catalog": None, "error": str(exc)}

    canonical = [{k: m[k] for k in ("member", "label", "sha256", "label_sha256", "split")} for m in members]
    identity = canonical_json_sha256(canonical)
    return {"status": "ready", "yaml_sha256": sha256_file(yaml_path), "train_val_sha256": identity,
            "members": canonical, "test_split_rejected": False, "split_catalog": catalog_identity,
            "dataset_yaml": str(yaml_path), "portable_root": str(dataset_root),
            "annotations_path": str(annotations_path), "split_manifest_path": str(split_manifest_path)}


def git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False).stdout.strip()
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {"sha": run("rev-parse", "HEAD") or None,
            "branch": run("branch", "--show-current") or None,
            "dirty": bool(status),
            "dirty_scope": "tracked and untracked state; unrelated private names omitted"}


def implementation_identity(root: Path = CODE_ROOT) -> dict[str, str]:
    paths = (
        "src/deskcheck/training/e1_experiment.py",
        "src/deskcheck/training/dataset_layout.py",
        "scripts/preflight_e1.py",
        "src/deskcheck/training/trainer.py",
        "src/deskcheck/training/training_config.py",
        "src/deskcheck/data/dataset_catalog.py",
        "docs/E1_protocol.md",
        "configs/experiments/baseline.json",
        "configs/experiments/augmentation.json",
    )
    return {path: sha256_file(root / path) for path in paths}


def _environment_identity() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("ultralytics", "torch", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def resolve_plan(control_path: Path, treatment_path: Path, root: Path | None = None,
                 baseline_path: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    control_path = (root / control_path).resolve() if not control_path.is_absolute() else control_path.resolve()
    treatment_path = (root / treatment_path).resolve() if not treatment_path.is_absolute() else treatment_path.resolve()
    baseline_path = baseline_path or root / "configs/experiments/baseline.json"
    baseline_path = (root / baseline_path).resolve() if not baseline_path.is_absolute() else baseline_path.resolve()

    load_config(baseline_path)
    control = load_config(control_path, baseline_path=baseline_path)
    treatment = load_config(treatment_path, baseline_path=baseline_path)

    allowed: dict[str, dict[str, Any]] = {}
    forbidden: dict[str, dict[str, Any]] = {}
    no_ops: list[str] = []

    for block_name in ("model", "data", "preprocessing", "augmentation", "training", "validation", "inference", "outputs"):
        control_block = getattr(control, block_name)
        treatment_block = getattr(treatment, block_name)
        for key in sorted(set(control_block) | set(treatment_block)):
            cv, tv = control_block.get(key), treatment_block.get(key)
            if cv == tv:
                if block_name == "augmentation" and key in SIX_KEYS and cv == 0:
                    no_ops.append(f"augmentation.{key}")
                continue
            location = f"{block_name}.{key}" if block_name != "augmentation" else f"augmentation.{key}"
            if block_name == "augmentation" and key in SIX_KEYS:
                allowed[location] = {"control": cv, "treatment": tv}
            elif block_name == "augmentation":
                forbidden[location] = {"control": cv, "treatment": tv}
            elif block_name == "outputs" and key == "run_dir":
                allowed[location] = {"control": cv, "treatment": tv}
            else:
                forbidden[location] = {"control": cv, "treatment": tv}

    policy_ok = all(key in treatment.augmentation and treatment.augmentation[key] == 0 for key in SIX_KEYS)
    return {
        "control_path": str(control_path), "treatment_path": str(treatment_path),
        "baseline_path": str(baseline_path),
        "config_identity": {
            "control_sha256": sha256_file(control_path),
            "treatment_sha256": sha256_file(treatment_path),
            "baseline_sha256": sha256_file(baseline_path),
        },
        "control": {
            "model": control.model, "data": control.data, "preprocessing": control.preprocessing,
            "augmentation": control.augmentation, "training": control.training,
            "validation": control.validation, "inference": control.inference, "outputs": control.outputs,
        },
        "treatment": {
            "model": treatment.model, "data": treatment.data, "preprocessing": treatment.preprocessing,
            "augmentation": treatment.augmentation, "training": treatment.training,
            "validation": treatment.validation, "inference": treatment.inference, "outputs": treatment.outputs,
        },
        "allowed_differences": allowed,
        "forbidden_differences": forbidden,
        "no_op_augmentation_keys": sorted(no_ops),
        "six_zero_policy_valid": policy_ok,
    }


def build_receipt(root: Path, plan: dict[str, Any], initialization: Path | None,
                  dataset_yaml: Path | None, receipt_dir: Path,
                  technical_ready: bool | None = None,
                  annotations_path: Path | None = None,
                  split_manifest_path: Path | None = None,
                  run_dir: Path | None = None,
                  approval_records: dict[str, bool] | None = None) -> dict[str, Any]:
    init = hash_initialization(initialization)
    inv = inventory_dataset(dataset_yaml, root, annotations_path, split_manifest_path)
    missing: list[str] = []
    if plan["forbidden_differences"]:
        missing.append("forbidden semantic differences")
    if not plan["six_zero_policy_valid"]:
        missing.append("treatment does not implement exact six-zero policy")
    if init["status"] != "present":
        missing.append("initialization bytes/identity missing")
    if inv["status"] != "ready":
        missing.append("effective train/val inventory or authoritative split membership incomplete")
    if run_dir is None:
        missing.append("execution output directory missing")
        frozen_run_dir = None
    else:
        raw_run_dir = Path(run_dir).expanduser()
        if raw_run_dir.is_symlink() or raw_run_dir.exists():
            missing.append("execution output directory must be new and nonexistent")
            frozen_run_dir = None
        else:
            frozen_run_dir = str(raw_run_dir.resolve())
            receipt_path = receipt_dir.expanduser().resolve()
            run_path = Path(frozen_run_dir)
            if (
                run_path == receipt_path
                or run_path in receipt_path.parents
                or receipt_path in run_path.parents
            ):
                missing.append("execution output directory must be separate from receipt output")
                frozen_run_dir = None

    approvals = {name: (approval_records or {}).get(name) is True for name in APPROVAL_FIELDS}
    approval_missing = [name for name, approved in approvals.items() if not approved]
    source_identity = implementation_identity()
    environment = _environment_identity()
    for package in ("ultralytics", "torch"):
        if environment["packages"][package] is None:
            missing.append(f"required runtime package missing: {package}")
    technical_ready = not missing if technical_ready is None else bool(technical_ready and not missing)
    formal_training_approved = technical_ready and not approval_missing
    frozen_spec = {
        "schema_version": SPECIFICATION_SCHEMA_VERSION,
        "plan": plan,
        "initialization": init,
        "dataset_yaml_path": inv.get("dataset_yaml") if dataset_yaml is not None else None,
        "annotations_path": inv.get("annotations_path") if inv.get("status") == "ready" else None,
        "split_manifest_path": inv.get("split_manifest_path") if inv.get("status") == "ready" else None,
        "effective_train_val": inv,
        "run_dir": frozen_run_dir,
        "source_identity": source_identity,
        "environment": environment,
        "approval_records": approvals,
        "technical_ready": technical_ready,
        "formal_training_approved": formal_training_approved,
        "missing_prerequisites": sorted(set(missing + approval_missing)),
        "execution_intent": False,
        "execution_permission": False,
    }
    spec_hash = canonical_json_sha256(frozen_spec)

    return {
        "receipt_version": RECEIPT_VERSION,
        "training_performed": False,
        "experiment_id": "E1",
        "technical_ready": technical_ready,
        "owner_approval": approvals["owner_scientific_approval"],
        "execution_intent": False,
        "execution_permission": False,
        "formal_training_approved": formal_training_approved,
        "source": git_identity(root),
        "implementation_identity": source_identity,
        "source_file_hashes": {
            str(path): sha256_file(root / path)
            for path in ("configs/experiments/augmentation.json", "configs/experiments/baseline.json", "docs/E1_protocol.md")
        },
        "plan": plan,
        "execution_specification": frozen_spec,
        "execution_specification_sha256": spec_hash,
        "initialization": init,
        "effective_train_val": inv,
        "environment": frozen_spec["environment"],
        "intended_output_locations": {
            "control": str((root / plan["control"]["outputs"]["run_dir"]).resolve()),
            "treatment": str((root / plan["treatment"]["outputs"]["run_dir"]).resolve()),
            "execution": frozen_run_dir,
        },
        "missing_prerequisites": frozen_spec["missing_prerequisites"],
        "readiness": (
            "not_ready" if not technical_ready
            else "approval_incomplete" if not formal_training_approved
            else "ready_for_explicit_execution_intent"
        ),
        "receipt_dir": str(receipt_dir.resolve()),
    }


def write_receipt(receipt: dict[str, Any], output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new receipt directory: {output}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "execution_specification.json").write_text(
        json.dumps(receipt["execution_specification"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def execute_e1(specification_path: Path, *, approval_token: str,
               trainer: Callable[..., Any] | None = None) -> Any:
    """Execute only from a persisted, hash-checked specification."""
    spec_path = specification_path.expanduser().resolve(strict=False)
    if not spec_path.is_file():
        raise RuntimeError(f"Saved execution specification is missing: {spec_path}")
    specification = _load_json_object(spec_path, "execution specification")
    _validate_specification(specification)

    receipt_path = spec_path.parent / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("Saved preflight receipt is missing")
    receipt = _load_json_object(receipt_path, "preflight receipt")
    _validate_receipt(receipt)
    if receipt["execution_specification_sha256"] != canonical_json_sha256(specification):
        raise RuntimeError("Saved execution specification hash mismatch")
    if receipt["execution_specification"] != specification:
        raise RuntimeError("Receipt embedded execution specification differs from saved specification")

    receipt_dir = receipt_path.parent.resolve(strict=False)
    if receipt["receipt_dir"] != str(receipt_dir):
        raise RuntimeError("Saved preflight receipt directory differs from its current location")
    expected_receipt_fields = {
        "owner_approval": specification["approval_records"]["owner_scientific_approval"],
        "technical_ready": specification["technical_ready"],
        "formal_training_approved": specification["formal_training_approved"],
        "execution_intent": specification["execution_intent"],
        "execution_permission": specification["execution_permission"],
        "plan": specification["plan"],
        "implementation_identity": specification["source_identity"],
        "initialization": specification["initialization"],
        "effective_train_val": specification["effective_train_val"],
        "environment": specification["environment"],
        "missing_prerequisites": specification["missing_prerequisites"],
    }
    if any(receipt[key] != value for key, value in expected_receipt_fields.items()):
        raise RuntimeError("Saved preflight receipt contradicts its execution specification")
    if receipt["intended_output_locations"]["execution"] != specification["run_dir"]:
        raise RuntimeError("Saved preflight receipt execution path differs from its specification")
    expected_readiness = (
        "not_ready" if not specification["technical_ready"]
        else "approval_incomplete" if not specification["formal_training_approved"]
        else "ready_for_explicit_execution_intent"
    )
    if receipt["readiness"] != expected_readiness:
        raise RuntimeError("Saved preflight receipt readiness differs from its specification")

    if specification["technical_ready"] is not True:
        raise RuntimeError("Saved specification is not technically ready")
    approvals = specification["approval_records"]
    if any(approvals[name] is not True for name in APPROVAL_FIELDS):
        raise RuntimeError("Saved specification is missing required scientific approval records")
    if specification["formal_training_approved"] is not True:
        raise RuntimeError("Formal training approval is not recorded in the saved specification")
    if approval_token != APPROVAL_TOKEN:
        raise RuntimeError("Explicit execution-intent approval token required")

    plan = specification["plan"]
    baseline_path = Path(plan["baseline_path"])
    treatment_path = Path(plan["treatment_path"])
    control_path = Path(plan["control_path"])
    actual_identity = {
        "control_sha256": sha256_file(control_path),
        "treatment_sha256": sha256_file(treatment_path),
        "baseline_sha256": sha256_file(baseline_path),
    }
    if actual_identity != plan["config_identity"]:
        raise RuntimeError("Saved specification is stale: config bytes changed")
    fresh = resolve_plan(control_path, treatment_path, CODE_ROOT, baseline_path)
    if fresh != plan:
        raise RuntimeError("Saved specification is stale: resolved settings changed")
    if fresh["forbidden_differences"] or not fresh["six_zero_policy_valid"]:
        raise RuntimeError("Plan has forbidden differences or invalid policy")

    if implementation_identity() != specification["source_identity"]:
        raise RuntimeError("Saved specification source identity differs from current E1 implementation")
    if _environment_identity() != specification["environment"]:
        raise RuntimeError("Current Python/package environment differs from the saved specification")
    initialization = specification["initialization"]
    init_path = Path(initialization["path"])
    init_now = hash_initialization(init_path)
    if init_now != initialization or init_now.get("status") != "present":
        raise RuntimeError("Initialization differs from the frozen specification")
    dataset_yaml = Path(specification["dataset_yaml_path"])
    annotations_path = Path(specification["annotations_path"])
    split_manifest_path = Path(specification["split_manifest_path"])
    inventory_now = inventory_dataset(
        dataset_yaml, CODE_ROOT, annotations_path, split_manifest_path
    )
    if inventory_now != specification["effective_train_val"] or inventory_now.get("status") != "ready":
        raise RuntimeError("Train/val dataset or authoritative split membership differs from specification")

    if trainer is None:
        from deskcheck.training.trainer import train as trainer
    run_dir_value = specification["run_dir"]
    if run_dir_value is None:
        raise RuntimeError("Saved specification has no frozen execution directory")
    run_dir = _validate_execution_directory(run_dir_value, receipt_dir)
    return trainer(
        treatment_path,
        run_dir=run_dir,
        dataset_yaml_override=dataset_yaml,
        baseline_config_path=baseline_path,
        initialization_override=init_path,
        expected_initialization_sha256=initialization["sha256"],
    )
