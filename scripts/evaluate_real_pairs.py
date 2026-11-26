"""Part C bridge: evaluate real photo pairs with B's authoritative EvaluationRunner.

Requires A's existing checkpoint and a reviewed B-format event manifest.
No training, auto-download, fake fallback, metric duplication or overwrite.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.real_settings import RealSettings
from deskcheck.evaluation.evaluation_runner import EvaluationRunner
from deskcheck.evaluation.failure_record import build_rerun_link, new_run_id

REQUIRED = {
    "schema_version",
    "dataset_release",
    "split",
    "pairs",
    "ground_truth",
    "eligible_no_change_pair_ids",
}
OPTIONAL_SOURCE_IDENTITIES = {
    "source_annotations_sha256",
    "source_split_manifest_sha256",
}
EVALUATION_PACKAGES = ("numpy", "Pillow", "ultralytics", "torch")


def _source_identity() -> dict:
    try:
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        raise RuntimeError(f"Could not capture evaluation source identity: {detail}") from error
    return {
        "source_revision": revision,
        "source_dirty": bool(status),
        "source_dirty_paths": sorted(line[3:] for line in status),
    }


def _package_versions() -> dict:
    versions = {}
    for package in EVALUATION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate evaluation manifest field: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token):
    raise ValueError(f"Nonfinite JSON values are invalid: {token}")


def load_manifest(source: Path):
    """Parse a reviewed B event fixture, preserving all pair IDs and image paths.

    Truth is returned only to EvaluationRunner; detector sees pixels, not labels.
    A's original pairs.jsonl needs an approved annotation conversion beforehand.
    """
    content = source.read_bytes()
    data = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if (
        not isinstance(data, dict)
        or not REQUIRED <= set(data)
        or set(data) - REQUIRED - OPTIONAL_SOURCE_IDENTITIES
        or bool(set(data) & OPTIONAL_SOURCE_IDENTITIES)
        != (set(data) & OPTIONAL_SOURCE_IDENTITIES == OPTIONAL_SOURCE_IDENTITIES)
    ):
        raise ValueError("Evaluation manifest has missing or unexpected fields")
    if data["schema_version"] != "1.0" or data["split"] not in ("val", "test"):
        raise ValueError("Evaluation manifest requires schema 1.0 and explicit val/test split")
    if not isinstance(data["dataset_release"], str) or not data["dataset_release"].strip():
        raise ValueError("Evaluation manifest requires a named dataset release")
    pairs = data["pairs"]
    truth = data["ground_truth"]
    eligible = data["eligible_no_change_pair_ids"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Evaluation manifest requires nonempty pairs list")
    if not isinstance(truth, dict) or not isinstance(eligible, list):
        raise TypeError("Evaluation manifest needs ground truth and no-change IDs")
    if any(not isinstance(x, str) or not x.strip() for x in eligible):
        raise ValueError("No-change pair IDs must be strings")
    ids = set()
    validated_pairs = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"pair_id", "before_path", "after_path"}:
            raise ValueError("Pair fields must be pair_id, before_path, after_path")
        pair_id = pair["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.strip() or pair_id in ids:
            raise ValueError("Pair IDs must be unique, nonempty strings")
        ids.add(pair_id)
        item = {"pair_id": pair_id}
        for field in ("before_path", "after_path"):
            name = pair[field]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{field} must be a nonempty image path")
            local = Path(name)
            if not local.is_absolute():
                local = source.parent / local
            local = local.resolve(strict=True)
            if not local.is_file():
                raise ValueError(f"Image source is not a file: {local}")
            item[field] = local
            digest = hashlib.sha256()
            with local.open("rb") as image_file:
                for block in iter(lambda: image_file.read(1024 * 1024), b""):
                    digest.update(block)
            item[field.replace("_path", "_sha256")] = digest.hexdigest()
        validated_pairs.append(item)
    if set(truth) != ids or not set(eligible).issubset(ids) or len(set(eligible)) != len(eligible):
        raise ValueError("Ground truth / no-change IDs must match the complete pair inventory")
    if any(not isinstance(events, list) for events in truth.values()):
        raise ValueError("Ground truth must contain explicit event lists")
    source_identities = {}
    for key in OPTIONAL_SOURCE_IDENTITIES:
        digest = data.get(key)
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
        source_identities[key] = digest
    return {
        "pairs": validated_pairs,
        "ground_truth": truth,
        "eligible_no_change_pair_ids": eligible,
        "split": data["split"],
        "dataset_release": data["dataset_release"],
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        **source_identities,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Part C real batch eval through B's runner")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=None,
        help="New directory for a failure record; defaults beside --output",
    )
    parser.add_argument(
        "--rerun-of",
        type=Path,
        action="append",
        default=[],
        help="Earlier failure_record.json this run replaces; repeatable",
    )
    parser.add_argument("--rerun-reason", default=None, help="Free-text reason for the rerun")
    args = parser.parse_args()
    target = args.output.absolute()
    if target.exists() or target.is_symlink():
        print(f"Refusing to overwrite evaluation: {target}", file=sys.stderr)
        return 2
    if args.rerun_of and not (args.rerun_reason and args.rerun_reason.strip()):
        print("--rerun-reason is required with --rerun-of", file=sys.stderr)
        return 2
    failure_output = args.failure_output or target.with_name(
        f"{target.name}.failure-{new_run_id()}"
    )
    try:
        rerun_of = [build_rerun_link(path, args.rerun_reason) for path in args.rerun_of]
    except (OSError, ValueError, TypeError) as error:
        print(f"Rerun link is invalid: {error}", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest.resolve(strict=True))
        settings = RealSettings.load(args.config)
        service = ComparisonService.for_real(settings)
        provenance = {
            "mode": "real",
            "run_id": (
                f"{manifest['split']}-{target.name}-{settings.checkpoint_sha256[:12]}-"
                f"{new_run_id()[:8]}"
            ),
            "config_sha256": settings.config_sha256,
            "checkpoint_sha256": settings.checkpoint_sha256,
            "model_origin": settings.model_origin,
            "rules_version": settings.rules_version,
            "confidence_threshold": settings.confidence_threshold,
            "movement_threshold": settings.movement_threshold,
            "nms_iou_threshold": settings.nms_iou_threshold,
            "imgsz": settings.imgsz,
            "split": manifest["split"],
            "dataset_release": manifest["dataset_release"],
            "manifest_sha256": manifest["manifest_sha256"],
            "source_annotations_sha256": manifest["source_annotations_sha256"],
            "source_split_manifest_sha256": manifest["source_split_manifest_sha256"],
            "python_version": platform.python_version(),
            "package_versions": _package_versions(),
            "command": sys.argv,
            **_source_identity(),
        }
        result = EvaluationRunner(service).run(
            pairs=manifest["pairs"],
            ground_truth=manifest["ground_truth"],
            eligible_no_change_pair_ids=manifest["eligible_no_change_pair_ids"],
            output_dir=target,
            metadata=provenance,
            failure_output_dir=failure_output,
            rerun_of=rerun_of,
        )
        print(
            f"mode=real split={manifest['split']} pairs={result['run_metadata']['pair_count']} "
            f"checkpoint={settings.checkpoint_sha256} output={target}; "
            "metric scope: B event/no-change only, not detector AP"
        )
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, UnicodeError) as error:
        print(f"Real batch evaluation failed without fake fallback: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
