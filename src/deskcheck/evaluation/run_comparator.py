"""Compare completed event-evaluation runs under identical frozen inputs/rules."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from deskcheck.evaluation.event_evaluator import (
    PRIMARY_METRIC_NAME,
    EventEvaluator,
)
from deskcheck.evaluation.failure_record import validate_rerun_links
from deskcheck.evaluation.no_change_evaluator import NoChangeEvaluator

_PRIMARY_ROLE = "primary"
_SECONDARY_ROLE = "secondary"

_COMMON_PROVENANCE = (
    "split",
    "dataset_release",
    "manifest_sha256",
    "source_annotations_sha256",
    "source_split_manifest_sha256",
    "confidence_threshold",
    "movement_threshold",
    "nms_iou_threshold",
    "imgsz",
    "rules_version",
    "python_version",
    "source_revision",
    "source_dirty",
    "source_dirty_paths",
    "package_versions",
)
_HASH = re.compile(r"[0-9a-f]{64}")
_EVENT_METRICS = ("tp", "fp", "fn", "support", "precision", "recall", "f1")


def _read_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON field {key!r} in {path}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value is invalid: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _required(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where} is missing required field {key!r}")
    return mapping[key]


def _finite_number(value: Any, where: str) -> float | int:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    return value


def _load_run(run_dir: Path, label: str) -> dict:
    if not run_dir.is_dir():
        raise ValueError(f"{label} evaluation directory does not exist: {run_dir}")
    metadata = _read_json(run_dir / "run_metadata.json")
    metrics = _read_json(run_dir / "metrics.json")
    predictions = _read_json(run_dir / "predictions.json")
    ground_truth = _read_json(run_dir / "ground_truth.json")
    if not all(isinstance(value, dict) for value in (metadata, metrics, predictions, ground_truth)):
        raise ValueError(f"{label} run artifacts must contain JSON objects")
    if metadata.get("schema_version") != "1.0" or metadata.get("status") != "complete":
        raise ValueError(f"{label} run is not a completed supported evaluation")
    if metadata.get("runner_protocol") != "event-runner-v1":
        raise ValueError(f"{label} run uses an unsupported evaluation runner protocol")
    provenance = _required(metadata, "provenance", f"{label} run metadata")
    if not isinstance(provenance, dict):
        raise TypeError(f"{label} provenance must be an object")

    run_id = _required(provenance, "run_id", f"{label} provenance")
    model_origin = _required(provenance, "model_origin", f"{label} provenance")
    checkpoint = _required(provenance, "checkpoint_sha256", f"{label} provenance")
    config_hash = _required(provenance, "config_sha256", f"{label} provenance")
    if "run_id" in metadata and metadata["run_id"] != run_id:
        raise ValueError(f"{label} run_id disagrees with its provenance run_id")
    for field, value in (
        ("run_id", run_id),
        ("model_origin", model_origin),
        ("checkpoint_sha256", checkpoint),
        ("config_sha256", config_hash),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} provenance.{field} must be nonempty")
    for field in ("checkpoint_sha256", "config_sha256", "manifest_sha256"):
        value = _required(provenance, field, f"{label} provenance")
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"{label} provenance.{field} must be a lowercase SHA-256")
    for field in ("source_annotations_sha256", "source_split_manifest_sha256"):
        value = _required(provenance, field, f"{label} provenance")
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"{label} provenance.{field} must be a lowercase SHA-256")

    for field in _COMMON_PROVENANCE:
        _required(provenance, field, f"{label} provenance")
    if provenance["split"] not in ("val", "test"):
        raise ValueError(f"{label} split must be val or test")
    if type(provenance["source_dirty"]) is not bool:
        raise ValueError(f"{label} source_dirty must be a boolean")
    if not isinstance(provenance["source_dirty_paths"], list) or any(
        not isinstance(path, str) for path in provenance["source_dirty_paths"]
    ):
        raise ValueError(f"{label} source_dirty_paths must be a list of strings")
    if provenance["source_dirty"] != bool(provenance["source_dirty_paths"]):
        raise ValueError(f"{label} dirty-state flag disagrees with dirty paths")
    if not isinstance(provenance["package_versions"], dict):
        raise TypeError(f"{label} package_versions must be an object")
    if any(
        not isinstance(version, str) and version is not None
        for version in provenance["package_versions"].values()
    ):
        raise ValueError(f"{label} package versions must be strings or null")
    for setting in (
        "confidence_threshold",
        "movement_threshold",
        "nms_iou_threshold",
    ):
        _finite_number(provenance[setting], f"{label} provenance.{setting}")
    if type(provenance["imgsz"]) is not int or provenance["imgsz"] <= 0:
        raise ValueError(f"{label} provenance.imgsz must be a positive integer")
    for field in ("dataset_release", "rules_version", "python_version", "source_revision"):
        if not isinstance(provenance[field], str) or not provenance[field].strip():
            raise ValueError(f"{label} provenance.{field} must be nonempty")

    pairs = metadata.get("pairs")
    if not isinstance(pairs, list):
        raise TypeError(f"{label} run metadata pairs must be a list")
    pair_ids = []
    for pair in pairs:
        expected_pair_fields = {
            "pair_id",
            "before_path",
            "after_path",
            "before_sha256",
            "after_sha256",
        }
        if not isinstance(pair, dict) or set(pair) != expected_pair_fields:
            raise ValueError(f"{label} pair inventory has an invalid record")
        if any(not isinstance(pair[key], str) or not pair[key] for key in pair):
            raise ValueError(f"{label} pair inventory fields must be nonempty strings")
        for key in ("before_sha256", "after_sha256"):
            if not _HASH.fullmatch(pair[key]):
                raise ValueError(f"{label} pair inventory {key} must be a lowercase SHA-256")
        pair_ids.append(pair["pair_id"])
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"{label} pair inventory must be nonempty with unique IDs")
    if metadata.get("pair_count") != len(pair_ids):
        raise ValueError(f"{label} pair_count disagrees with pair inventory")
    expected_ids = set(pair_ids)
    if set(predictions) != expected_ids or set(ground_truth) != expected_ids:
        raise ValueError(f"{label} predictions/ground truth do not match complete pair inventory")

    event_metrics = _required(metrics, "events", f"{label} metrics")
    no_change_metrics = _required(metrics, "no_change", f"{label} metrics")
    if not isinstance(event_metrics, dict) or not isinstance(no_change_metrics, dict):
        raise TypeError(f"{label} event and no-change metrics must be objects")
    if event_metrics.get("protocol_version") != "basic-localized-events-v1":
        raise ValueError(f"{label} event metric protocol is unsupported")
    if event_metrics.get("zero_division") != 0:
        raise ValueError(f"{label} event metric zero-division rule is unsupported")
    _finite_number(event_metrics.get("iou_threshold"), f"{label} event IoU threshold")
    if event_metrics.get("averaging") != "fixed status set; absent-status F1 is zero":
        raise ValueError(f"{label} event averaging rule is unsupported")
    per_status = _required(event_metrics, "per_status", f"{label} event metrics")
    per_pair = _required(event_metrics, "per_pair", f"{label} event metrics")
    if not isinstance(per_status, dict) or not isinstance(per_pair, dict):
        raise TypeError(f"{label} per-status and per-pair metrics must be objects")
    if set(per_pair) != expected_ids:
        raise ValueError(f"{label} per-pair metrics do not match complete pair inventory")
    for status_values in per_status.values():
        if not isinstance(status_values, dict):
            raise TypeError(f"{label} per-status metric must be an object")
        for key in _EVENT_METRICS:
            _finite_number(_required(status_values, key, f"{label} per-status metric"), key)
    for pair_id, pair_values in per_pair.items():
        if not isinstance(pair_values, dict) or not isinstance(pair_values.get("per_status"), dict):
            raise TypeError(f"{label} per-pair metrics are malformed for {pair_id!r}")

    if no_change_metrics.get("protocol_version") != "independent-no-change-v1":
        raise ValueError(f"{label} no-change metric protocol is unsupported")
    eligible = metadata.get("eligible_no_change_pair_ids")
    if (
        not isinstance(eligible, list)
        or any(not isinstance(pair_id, str) or pair_id not in expected_ids for pair_id in eligible)
        or len(eligible) != len(set(eligible))
    ):
        raise ValueError(f"{label} eligible no-change pair IDs are invalid")
    for key in (
        "eligible_pair_count",
        "false_positive_event_count",
        "false_alarm_pair_count",
    ):
        value = _required(no_change_metrics, key, f"{label} no-change metrics")
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} no-change metric {key} must be a nonnegative integer")
    _finite_number(
        _required(no_change_metrics, "false_alarm_pair_rate", f"{label} no-change metrics"),
        f"{label} false alarm pair rate",
    )
    if no_change_metrics["eligible_pair_count"] != len(eligible):
        raise ValueError(f"{label} no-change denominator differs from eligible pair IDs")
    if no_change_metrics.get("false_alarm_rate_unit") != (
        "pairs with >=1 predicted change / eligible no-change pairs"
    ):
        raise ValueError(f"{label} false-alarm rate does not declare its unit and denominator")
    if "primary_metric" not in event_metrics:
        raise ValueError(f"{label} run does not record the primary downstream metric")
    primary = event_metrics["primary_metric"]
    if not isinstance(primary, dict):
        raise TypeError(f"{label} run primary metric must be an object")
    if primary.get("name") != PRIMARY_METRIC_NAME or primary.get("role") != _PRIMARY_ROLE:
        raise ValueError(f"{label} run primary metric is not {PRIMARY_METRIC_NAME}")
    if primary.get("value") != event_metrics["change_only_macro_f1"]:
        raise ValueError(f"{label} run primary metric value disagrees with its scored metric")
    recorded_primary = metadata.get("primary_downstream_metric")
    if not isinstance(recorded_primary, dict) or (
        recorded_primary.get("name") != PRIMARY_METRIC_NAME
        or recorded_primary.get("role") != _PRIMARY_ROLE
        or recorded_primary.get("value") != event_metrics["change_only_macro_f1"]
    ):
        raise ValueError(f"{label} run metadata does not agree with the primary metric")
    calculated_events = EventEvaluator(event_metrics["iou_threshold"]).evaluate(
        predictions, ground_truth
    )
    if calculated_events != event_metrics:
        raise ValueError(f"{label} saved event metrics do not match its saved predictions")
    calculated_no_change = NoChangeEvaluator().evaluate(predictions, eligible)
    if calculated_no_change != no_change_metrics:
        raise ValueError(f"{label} saved no-change metrics do not match its saved predictions")
    if metadata.get("event_iou_threshold") != event_metrics["iou_threshold"]:
        raise ValueError(f"{label} run metadata event IoU threshold differs from its metrics")
    rerun_of = validate_rerun_links(metadata.get("rerun_of", []))

    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "provenance": provenance,
        "metrics": metrics,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "pair_ids": pair_ids,
        "eligible": eligible,
        "run_id": run_id,
        "rerun_of": rerun_of,
    }


def _metric_comparison(control: Any, treatment: Any, role: str) -> dict:
    if type(control) not in (int, float) or type(treatment) not in (int, float):
        raise ValueError("Comparable metric values must both be numeric")
    if not math.isfinite(control) or not math.isfinite(treatment):
        raise ValueError("Comparable metric values must be finite")
    return {
        "role": role,
        "control": control,
        "treatment": treatment,
        "treatment_minus_control": treatment - control,
    }


def _event_comparison(control: dict, treatment: dict) -> dict:
    return {
        metric: _metric_comparison(control[metric], treatment[metric], _SECONDARY_ROLE)
        for metric in _EVENT_METRICS
    }


def compare_runs(control_dir: str | Path, treatment_dir: str | Path) -> dict:
    """Compare two saved downstream runs without loading a model or image."""
    control = _load_run(Path(control_dir), "control")
    treatment = _load_run(Path(treatment_dir), "treatment")
    cp, tp = control["provenance"], treatment["provenance"]

    for field in _COMMON_PROVENANCE:
        if cp[field] != tp[field]:
            raise ValueError(f"Evaluation runs differ in frozen {field}")
    if cp["run_id"] == tp["run_id"]:
        raise ValueError("Control and treatment must have distinct run IDs")
    if cp["checkpoint_sha256"] == tp["checkpoint_sha256"]:
        raise ValueError("Control and treatment must identify distinct checkpoint bytes")
    if control["metadata"]["event_iou_threshold"] != treatment["metadata"]["event_iou_threshold"]:
        raise ValueError("Evaluation runs differ in event IoU threshold")
    if control["metadata"]["event_iou_threshold"] != control["metrics"]["events"]["iou_threshold"]:
        raise ValueError("Control event IoU metadata disagrees with scored metrics")
    if (
        treatment["metadata"]["event_iou_threshold"]
        != treatment["metrics"]["events"]["iou_threshold"]
    ):
        raise ValueError("Treatment event IoU metadata disagrees with scored metrics")
    if control["pair_ids"] != treatment["pair_ids"]:
        raise ValueError("Evaluation runs have different ordered pair inventories")
    if control["metadata"]["pairs"] != treatment["metadata"]["pairs"]:
        raise ValueError("Evaluation runs do not refer to the same image paths")
    if control["eligible"] != treatment["eligible"]:
        raise ValueError("Evaluation runs have different no-change eligibility")
    if control["ground_truth"] != treatment["ground_truth"]:
        raise ValueError("Evaluation runs have different ground truth")
    if (
        control["metrics"]["events"]["per_status"].keys()
        != treatment["metrics"]["events"]["per_status"].keys()
    ):
        raise ValueError("Evaluation runs have different event status sets")
    if (
        control["metrics"]["events"]["per_pair"].keys()
        != treatment["metrics"]["events"]["per_pair"].keys()
    ):
        raise ValueError("Evaluation runs have different per-pair metric inventories")

    control_detection_path = control["run_dir"] / "image_detections.json"
    treatment_detection_path = treatment["run_dir"] / "image_detections.json"
    if control_detection_path.exists() != treatment_detection_path.exists():
        raise ValueError("Only one evaluation run contains per-image detections")
    image_detections_available = control_detection_path.exists()
    if image_detections_available:
        control_detections = _read_json(control_detection_path)
        treatment_detections = _read_json(treatment_detection_path)
        if (
            not isinstance(control_detections, dict)
            or not isinstance(treatment_detections, dict)
            or set(control_detections) != set(control["pair_ids"])
            or set(treatment_detections) != set(treatment["pair_ids"])
        ):
            raise ValueError("Per-image detection evidence does not match the pair inventory")

    control_events = control["metrics"]["events"]
    treatment_events = treatment["metrics"]["events"]
    if (
        control_events["primary_metric"]["decision_date"]
        != treatment_events["primary_metric"]["decision_date"]
    ):
        raise ValueError("Evaluation runs disagree on the primary-metric decision date")
    primary_metric = _metric_comparison(
        control_events["change_only_macro_f1"],
        treatment_events["change_only_macro_f1"],
        _PRIMARY_ROLE,
    )
    primary_metric.update(
        {
            "name": PRIMARY_METRIC_NAME,
            "decision_date": control_events["primary_metric"]["decision_date"],
            "aggregation": control_events["primary_metric"]["aggregation"],
            "zero_division": 0,
        }
    )
    aggregate = {
        "event_metrics": {
            "change_only_macro_f1": dict(primary_metric),
            "macro_f1": _metric_comparison(
                control_events["macro_f1"], treatment_events["macro_f1"], _SECONDARY_ROLE
            ),
        },
        "per_status": {
            status: _event_comparison(
                control_events["per_status"][status],
                treatment_events["per_status"][status],
            )
            for status in sorted(control_events["per_status"])
        },
        "no_change": {
            metric: _metric_comparison(
                control["metrics"]["no_change"][metric],
                treatment["metrics"]["no_change"][metric],
                _SECONDARY_ROLE,
            )
            for metric in (
                "false_positive_event_count",
                "false_alarm_pair_count",
                "false_alarm_pair_rate",
            )
        },
    }
    per_pair = {}
    for pair_id in control["pair_ids"]:
        left = control_events["per_pair"][pair_id]
        right = treatment_events["per_pair"][pair_id]
        if left.keys() != right.keys() or left["per_status"].keys() != right["per_status"].keys():
            raise ValueError(f"Per-pair metrics differ in shape for {pair_id!r}")
        per_pair[pair_id] = {
            "event_metrics": {
                "change_only_macro_f1": _metric_comparison(
                    left["change_only_macro_f1"], right["change_only_macro_f1"], _PRIMARY_ROLE
                ),
                "macro_f1": _metric_comparison(
                    left["macro_f1"], right["macro_f1"], _SECONDARY_ROLE
                ),
            },
            "per_status": {
                status: _event_comparison(left["per_status"][status], right["per_status"][status])
                for status in sorted(left["per_status"])
            },
        }

    return {
        "schema_version": "1.0",
        "status": "complete",
        "comparison_scope": "downstream localized-event metrics; detector AP is not included",
        "split": cp["split"],
        "dataset_release": cp["dataset_release"],
        "manifest_sha256": cp["manifest_sha256"],
        "source_annotations_sha256": cp["source_annotations_sha256"],
        "source_split_manifest_sha256": cp["source_split_manifest_sha256"],
        "pair_count": len(control["pair_ids"]),
        "event_iou_threshold": control["metadata"]["event_iou_threshold"],
        "primary_metric": primary_metric,
        "metric_roles": {
            "primary": [PRIMARY_METRIC_NAME],
            "secondary": [
                "macro_f1",
                "per_status (all four statuses)",
                "no_change false-positive event count",
                "no_change pair-level false-alarm rate",
            ],
        },
        "primary_metric_note": (
            "Primary downstream metric is the change-only macro F1 (owner decision "
            "2026-11-26). Absent change statuses keep F1 = 0 under the fixed three-status "
            "denominator, so the value is defined rather than unavailable. All other "
            "metrics are secondary and no detector result is inferred from them."
        ),
        "common_evaluation_settings": {
            key: cp[key]
            for key in (
                "confidence_threshold",
                "movement_threshold",
                "nms_iou_threshold",
                "imgsz",
                "rules_version",
                "python_version",
                "source_revision",
                "source_dirty",
                "source_dirty_paths",
                "package_versions",
            )
        },
        "control": {
            "run_id": cp["run_id"],
            "model_origin": cp["model_origin"],
            "checkpoint_sha256": cp["checkpoint_sha256"],
            "config_sha256": cp["config_sha256"],
        },
        "treatment": {
            "run_id": tp["run_id"],
            "model_origin": tp["model_origin"],
            "checkpoint_sha256": tp["checkpoint_sha256"],
            "config_sha256": tp["config_sha256"],
        },
        "image_detections_available": image_detections_available,
        "linked_failures": {
            "control": control["rerun_of"],
            "treatment": treatment["rerun_of"],
            "note": (
                "Rerun links were re-validated before comparison. Whether a rerun replaces "
                "a failed run is an owner decision; this comparison neither accepts nor "
                "rejects it."
            ),
        },
        "aggregate": aggregate,
        "per_pair": per_pair,
    }


def render_markdown(report: dict) -> str:
    """Render a concise, human-readable downstream comparison summary."""
    primary = report["primary_metric"]
    lines = [
        "# E1 Downstream Evaluation Comparison",
        "",
        f"- Split: `{report['split']}`",
        f"- Dataset release: `{report['dataset_release']}`",
        f"- Pairs: {report['pair_count']}",
        f"- Event IoU threshold: {report['event_iou_threshold']}",
        f"- Primary metric: `{primary['name']}` (owner decision {primary['decision_date']})",
        "- Scope: downstream localized events only; detector AP is not included.",
        "",
        "| Metric | Role | Control | Treatment | Treatment - control |",
        "|---|---|---:|---:|---:|",
        (
            f"| **{primary['name']}** | **primary** | {primary['control']:.6g} "
            f"| {primary['treatment']:.6g} | {primary['treatment_minus_control']:+.6g} |"
        ),
    ]
    for metric, values in report["aggregate"]["event_metrics"].items():
        if metric == primary["name"]:
            continue
        lines.append(
            f"| {metric} | secondary | {values['control']:.6g} | {values['treatment']:.6g} "
            f"| {values['treatment_minus_control']:+.6g} |"
        )
    for status, metrics in report["aggregate"]["per_status"].items():
        for metric in ("tp", "fp", "fn", "precision", "recall", "f1"):
            values = metrics[metric]
            lines.append(
                f"| {status} {metric} | secondary | {values['control']:.6g} "
                f"| {values['treatment']:.6g} | {values['treatment_minus_control']:+.6g} |"
            )
    for metric, values in report["aggregate"]["no_change"].items():
        lines.append(
            f"| no-change {metric} | secondary | {values['control']:.6g} "
            f"| {values['treatment']:.6g} | {values['treatment_minus_control']:+.6g} |"
        )
    linked = report["linked_failures"]
    lines.extend(
        [
            "",
            (
                "The false-positive event count counts predicted added/removed/moved events "
                "on eligible no-change pairs. The false-alarm pair rate is a separate "
                "pair-level rate: pairs with at least one predicted change divided by "
                "eligible no-change pairs."
            ),
            "",
            (
                "Absent change statuses keep F1 = 0 under the fixed three-status "
                "denominator, so the primary value is defined rather than unavailable. "
                "No detector AP or test result is inferred from this downstream report."
            ),
            "",
            "Per-pair counts and deltas are available in `comparison.json`.",
            "",
            "## Linked failures",
            "",
        ]
    )
    for arm in ("control", "treatment"):
        links = linked[arm]
        if not links:
            lines.append(f"- {arm}: no rerun links recorded")
            continue
        for link in links:
            lines.append(
                f"- {arm}: failure `{link['failure_id']}` "
                f"(`{link['failure_record_path']}`) — rerun reason: {link['reason']}"
            )
    lines.extend(["", linked["note"], ""])
    return "\n".join(lines)


def write_comparison(report: dict, output_dir: str | Path) -> Path:
    """Write JSON and Markdown to a new directory without replacing prior evidence."""
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite comparison output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / "comparison.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "summary.md").write_text(render_markdown(report), encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite comparison output: {target}")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "comparison.json"
