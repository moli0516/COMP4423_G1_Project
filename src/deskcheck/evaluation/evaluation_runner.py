"""Complete-inventory event evaluation with an injected comparison service."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deskcheck.evaluation.event_evaluator import EventEvaluator
from deskcheck.evaluation.evidence_labels import IMAGE_DETECTION_EVIDENCE_LABEL
from deskcheck.evaluation.failure_record import (
    build_failure_record,
    new_run_id,
    validate_rerun_links,
    write_failure_record,
)
from deskcheck.evaluation.no_change_evaluator import NoChangeEvaluator

_FAILURE_CONFIG_FIELDS = (
    "confidence_threshold",
    "nms_iou_threshold",
    "movement_threshold",
    "imgsz",
    "rules_version",
)

_PRIMARY_METRIC_ROLES = {
    "change_only_macro_f1": "primary",
    "macro_f1": "secondary",
    "per_status": "secondary",
    "no_change": "secondary",
}


def _evaluation_config(provenance: dict, output_dir: object, iou_threshold: float) -> dict:
    config = {key: provenance[key] for key in _FAILURE_CONFIG_FIELDS if key in provenance}
    config["event_iou_threshold"] = iou_threshold
    config["output_dir"] = str(output_dir)
    return config


def _failure_record(
    *,
    exc: Exception,
    context: dict,
    metadata: Any,
    rerun_of: list[dict],
    run_id: str,
    output_dir: object,
    event_iou_threshold: float,
) -> dict:
    provenance = metadata if isinstance(metadata, dict) else {}
    return build_failure_record(
        run_id=run_id,
        reason=str(exc) or type(exc).__name__,
        exception_type=type(exc).__name__,
        context=context,
        input_identities={
            "manifest_sha256": provenance.get("manifest_sha256"),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "config_sha256": provenance.get("config_sha256"),
            "dataset_release": provenance.get("dataset_release"),
            "split": provenance.get("split"),
        },
        source_identity={
            "source_revision": provenance.get("source_revision"),
            "source_dirty": provenance.get("source_dirty"),
            "source_dirty_paths": provenance.get("source_dirty_paths"),
        },
        evaluation_config=_evaluation_config(provenance, output_dir, event_iou_threshold),
        command=provenance.get("command"),
        rerun_of=rerun_of,
    )


class EvaluationRunner:
    """Orchestrate inference and event metrics, not model loading or dataset parsing."""

    def __init__(self, comparison_service: object, iou_threshold: float = 0.5) -> None:
        compare = getattr(comparison_service, "compare", None)
        if not callable(compare):
            raise TypeError("comparison_service must expose a callable compare method")
        self._compare: Callable = compare
        self._events = EventEvaluator(iou_threshold=iou_threshold)
        self._no_change = NoChangeEvaluator()

    def run(
        self,
        pairs: list[dict],
        ground_truth: dict,
        eligible_no_change_pair_ids: list[str],
        output_dir: str | Path,
        metadata: dict | None = None,
        *,
        failure_output_dir: str | Path | None = None,
        rerun_of: list[dict] | None = None,
    ) -> dict:
        """Evaluate every pair and publish a new output directory only on success.

        Pair keys: pair_id, before_path, after_path, and optional paired SHA-256
        image hashes. Hashed files are checked immediately before and after
        compare(). compare() must return {"pair_id": same_id, "records": [event, ...]}.
        Caller metadata supplies actual split/model/config/rule identifiers; these
        are never invented here. Output directory must not already exist.

        If a run fails and ``failure_output_dir`` is given, one validated failure
        record is written there before the exception is re-raised; success-shaped
        files are never published for a failed run. ``rerun_of`` links the run to
        earlier failure records and is re-validated before any work starts.
        """
        rerun_links = validate_rerun_links(rerun_of) if rerun_of is not None else []
        context = {"stage": "preflight", "pair_id": None, "image_id": None}
        supplied_run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        if supplied_run_id is not None and (
            not isinstance(supplied_run_id, str) or not supplied_run_id.strip()
        ):
            raise ValueError("metadata run_id must be a nonempty string")
        run_id = supplied_run_id or new_run_id()
        try:
            return self._run(
                pairs,
                ground_truth,
                eligible_no_change_pair_ids,
                output_dir,
                metadata,
                run_id=run_id,
                rerun_of=rerun_links,
                context=context,
            )
        except Exception as exc:
            if failure_output_dir is None:
                raise
            try:
                record = _failure_record(
                    exc=exc,
                    context=context,
                    metadata=metadata,
                    rerun_of=rerun_links,
                    run_id=run_id,
                    output_dir=output_dir,
                    event_iou_threshold=self._events.iou_threshold,
                )
                write_failure_record(record, failure_output_dir)
            except Exception as record_exc:
                raise RuntimeError(
                    "Evaluation failed and its failure record could not be written"
                ) from record_exc
            raise

    def _run(
        self,
        pairs: list[dict],
        ground_truth: dict,
        eligible_no_change_pair_ids: list[str],
        output_dir: str | Path,
        metadata: dict | None,
        *,
        run_id: str,
        rerun_of: list[dict],
        context: dict,
    ) -> dict:
        normalized = self._normalize_pairs(pairs)
        truth = self._snapshot(ground_truth)
        # Public evaluator APIs validate ground truth before inference starts.
        self._events.evaluate(truth, truth)
        if set(truth) != {pair["pair_id"] for pair in normalized}:
            raise ValueError("Pair inventory and ground truth must match exactly")
        eligible = self._snapshot(eligible_no_change_pair_ids)
        eligibility = self._no_change.evaluate(truth, eligible)
        if eligibility["false_alarm_pair_count"]:
            raise ValueError("Eligible no-change ground truth contains change events")
        if any(not truth[pair_id] for pair_id in eligible):
            raise ValueError("Eligible no-change pairs require annotated unchanged objects")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")
        provenance = self._snapshot(metadata if metadata is not None else {})
        target = Path(output_dir)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite output directory: {target}")
        started = datetime.now(UTC).isoformat()
        predictions = {}
        image_detections = {}
        for pair in normalized:
            pair_id = pair["pair_id"]
            context["stage"] = "per_pair"
            context["pair_id"] = pair_id
            try:
                _verify_pair_hashes(pair)
                raw = self._compare(
                    before_path=pair["before_path"],
                    after_path=pair["after_path"],
                    pair_id=pair_id,
                )
                if not isinstance(raw, dict):
                    raise TypeError("compare must return a dictionary")
                if raw.get("pair_id") != pair_id:
                    raise ValueError("compare returned a missing or mismatched pair_id")
                if "records" not in raw:
                    raise ValueError("compare result is missing records")
                _verify_pair_hashes(pair)
                records = self._snapshot(raw["records"])
                if ("before_detections" in raw) != ("after_detections" in raw):
                    raise ValueError(
                        "compare result must provide detections for both images or neither"
                    )
                if "before_detections" in raw:
                    image_detections[pair_id] = {
                        "before": self._snapshot(raw["before_detections"]),
                        "after": self._snapshot(raw["after_detections"]),
                    }
                # Validate and snapshot immediately, even if the service reuses buffers.
                self._events.evaluate({pair_id: records}, {pair_id: truth[pair_id]})
                predictions[pair_id] = records
            except Exception as exc:
                raise RuntimeError(f"Evaluation failed for pair {pair_id!r}") from exc
        metrics = {
            "events": self._events.evaluate(predictions, truth),
            "no_change": self._no_change.evaluate(predictions, eligible),
        }
        if image_detections and len(image_detections) != len(normalized):
            raise RuntimeError(
                "Comparison service returned image detections for only part of the pair inventory"
            )
        run_metadata = {
            "schema_version": "1.0",
            "status": "complete",
            "runner_protocol": "event-runner-v1",
            "run_id": run_id,
            "started_at_utc": started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "pair_count": len(normalized),
            "pairs": [
                {
                    "pair_id": p["pair_id"],
                    "before_path": str(p["before_path"]),
                    "after_path": str(p["after_path"]),
                    **(
                        {
                            "before_sha256": p["before_sha256"],
                            "after_sha256": p["after_sha256"],
                        }
                        if "before_sha256" in p
                        else {}
                    ),
                }
                for p in normalized
            ],
            "eligible_no_change_pair_ids": sorted(eligible),
            "event_iou_threshold": metrics["events"]["iou_threshold"],
            "metric_scope": "localized events and no-change alarms; excludes detector AP",
            "image_detection_evidence": (
                IMAGE_DETECTION_EVIDENCE_LABEL if image_detections else "not provided"
            ),
            "primary_downstream_metric": {
                "name": metrics["events"]["primary_metric"]["name"],
                "role": "primary",
                "value": metrics["events"]["primary_metric"]["value"],
                "decision_date": metrics["events"]["primary_metric"]["decision_date"],
                "aggregation": metrics["events"]["primary_metric"]["aggregation"],
                "secondary_metrics": [
                    "macro_f1",
                    "per_status (all four statuses)",
                    "no_change false-positive event count and pair-level false-alarm rate",
                ],
            },
            "metric_roles": dict(_PRIMARY_METRIC_ROLES),
            "rerun_of": rerun_of,
            "provenance": provenance,
        }
        payloads = {
            "predictions.json": predictions,
            "ground_truth.json": truth,
            "metrics.json": metrics,
            "run_metadata.json": run_metadata,
        }
        if image_detections:
            payloads["image_detections.json"] = image_detections
        context["stage"] = "publish"
        context["pair_id"] = None
        self._publish(target, payloads)
        return {"predictions": predictions, "metrics": metrics, "run_metadata": run_metadata}

    @staticmethod
    def _normalize_pairs(pairs: list[dict]) -> list[dict]:
        if not isinstance(pairs, list):
            raise TypeError("pairs must be a list")
        if not pairs:
            raise ValueError("At least one pair is required")
        normalized = []
        seen = set()
        for pair in pairs:
            if not isinstance(pair, dict):
                raise TypeError("Each pair must be a dictionary")
            if not {"pair_id", "before_path", "after_path"} <= pair.keys():
                raise ValueError("Pair is missing required fields")
            pair_id = pair["pair_id"]
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError("pair_id must be a nonempty string")
            if pair_id in seen:
                raise ValueError(f"Duplicate pair_id: {pair_id}")
            seen.add(pair_id)
            item = {"pair_id": pair_id}
            for side in ("before_path", "after_path"):
                value = pair[side]
                if not isinstance(value, (str, Path)):
                    raise TypeError("Image paths must be strings or Path objects")
                if not str(value).strip():
                    raise ValueError("Image paths must not be empty")
                item[side] = Path(value)
            hashes = {}
            for side in ("before", "after"):
                key = f"{side}_sha256"
                if key in pair:
                    digest = pair[key]
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
                    hashes[key] = digest
            if bool(hashes) and set(hashes) != {"before_sha256", "after_sha256"}:
                raise ValueError("Image hashes must be provided for both sides or neither")
            item.update(hashes)
            normalized.append(item)
        return sorted(normalized, key=lambda pair: pair["pair_id"])

    @staticmethod
    def _snapshot(value: object) -> object:
        # Reject NaN/Infinity and unsupported objects rather than converting to strings.
        return json.loads(json.dumps(value, allow_nan=False))

    @staticmethod
    def _publish(target: Path, payloads: dict) -> None:
        serialized = {
            name: json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
            for name, value in payloads.items()
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            for name, text in serialized.items():
                (staging / name).write_text(text, encoding="utf-8")
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite output directory: {target}")
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_pair_hashes(pair: dict) -> None:
    for side in ("before", "after"):
        expected_sha = pair.get(f"{side}_sha256")
        if expected_sha is not None and _sha256_file(pair[f"{side}_path"]) != expected_sha:
            raise ValueError(f"{side} image bytes differ from the evaluation manifest")
