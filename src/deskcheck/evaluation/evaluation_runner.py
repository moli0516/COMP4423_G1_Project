"""Complete-inventory event evaluation with an injected comparison service."""

import json
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from deskcheck.evaluation.event_evaluator import EventEvaluator
from deskcheck.evaluation.no_change_evaluator import NoChangeEvaluator


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
    ) -> dict:
        """Evaluate every pair and publish a new output directory only on success.

        Pair keys: pair_id, before_path, after_path. Paths are passed as Path
        objects without reading files: the injected service owns image validation.
        compare() must return {"pair_id": same_id, "records": [event, ...]}.
        Caller metadata supplies actual split/model/config/rule identifiers; these
        are never invented here. Output directory must not already exist.
        """
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
        for pair in normalized:
            pair_id = pair["pair_id"]
            try:
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
                records = self._snapshot(raw["records"])
                # Validate and snapshot immediately, even if the service reuses buffers.
                self._events.evaluate({pair_id: records}, {pair_id: truth[pair_id]})
                predictions[pair_id] = records
            except Exception as exc:
                raise RuntimeError(f"Evaluation failed for pair {pair_id!r}") from exc
        metrics = {
            "events": self._events.evaluate(predictions, truth),
            "no_change": self._no_change.evaluate(predictions, eligible),
        }
        run_metadata = {
            "schema_version": "1.0",
            "status": "complete",
            "runner_protocol": "event-runner-v1",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "pair_count": len(normalized),
            "pairs": [
                {"pair_id": p["pair_id"], "before_path": str(p["before_path"]),
                 "after_path": str(p["after_path"])} for p in normalized
            ],
            "eligible_no_change_pair_ids": sorted(eligible),
            "event_iou_threshold": metrics["events"]["iou_threshold"],
            "metric_scope": "localized events and no-change alarms; excludes detector AP",
            "provenance": provenance,
        }
        payloads = {
            "predictions.json": predictions,
            "metrics.json": metrics,
            "run_metadata.json": run_metadata,
        }
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
