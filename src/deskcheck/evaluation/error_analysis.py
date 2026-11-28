"""Read-only error analysis for B event evaluation results.

Inputs are a `predictions.json` written by `EvaluationRunner` (`{pair_id:
[record, ...]}`, see `evaluation_runner.py:60-78,101-105`) and a B-format
manifest read by `scripts/evaluate_real_pairs.py:load_manifest`. The tool only
reports: it never changes thresholds, filters, rules or configuration, and it
must not be used to tune anything. It reads JSON only; it never opens, decodes,
resizes or stats an image (the mandated `load_manifest` stats manifest image
paths itself, unchanged).

Every accept/reject decision below is delegated to the existing evaluators:

- `EventEvaluator.evaluate` matches one event per class_id per pair and scores
  it as correct only when the status matches and every shared box side has IoU
  >= its threshold (`event_evaluator.py:19-70,127-133`). This module calls it
  once per (pair_id, class_id) on single-event inputs, so no second matching or
  IoU implementation exists here. The sum of those verdicts is checked against
  the evaluator's own aggregate counts before anything is written.
- `NoChangeEvaluator.evaluate` flags an eligible pair as a false alarm when any
  predicted event is not `unchanged` (`no_change_evaluator.py:29-60`).

Error types (only distinctions the evaluators can make):

- ``missed_event``: a ground-truth event has no prediction for that class_id;
  the evaluator counts one FN under the ground-truth status.
- ``spurious_event``: a prediction has no ground-truth counterpart for that
  class_id; the evaluator counts one FP under the predicted status.
- ``wrong_status``: both sides exist but the statuses differ; the evaluator
  counts one FP under the predicted status and one FN under the truth status.
- ``misaligned_event``: both sides exist with the same status but the
  evaluator's localization check fails; counted as FP + FN as above.
- ``other``: the evaluator's single-event verdict does not have the expected
  shape, so the case cannot be classified; the reason is recorded verbatim.

Wrong-class predictions are deliberately not a separate type: matching is keyed
by class_id only, so a wrong-class prediction appears as `missed_event` for the
true class plus `spurious_event` for the predicted class. No cross-class pairing
exists in the evaluator, so inventing a `wrong_class` type would be fiction.

No-change cases are reported per pair (the evaluator is pair-level) with
error_type ``none`` for a success and ``false_alarm`` for a failure.

Outputs are deterministic for identical input: rows are ordered by
(pair_id, class_id) and (pair_id), no timestamps or paths are embedded, and JSON
is written with sorted keys.

Stability limits: output is byte-identical for the same input under the same
Python version; float representation (for example 2/3) is not guaranteed to be
identical across Python versions. The event IoU threshold is supplied by the
caller and is not recorded in the predictions file, so it must be stated
explicitly on every run and compared runs must use the same value.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path

from deskcheck.evaluation.event_evaluator import EventEvaluator
from deskcheck.evaluation.no_change_evaluator import NoChangeEvaluator

DEFAULT_CASES_PER_GROUP = 2

MISSED_EVENT = "missed_event"
SPURIOUS_EVENT = "spurious_event"
WRONG_STATUS = "wrong_status"
MISALIGNED_EVENT = "misaligned_event"
OTHER = "other"

ERROR_TYPES: tuple[str, ...] = (
    MISALIGNED_EVENT,
    MISSED_EVENT,
    OTHER,
    SPURIOUS_EVENT,
    WRONG_STATUS,
)

PAIR_RESULT_FIELDS = (
    "pair_id",
    "is_eligible_nochange",
    "gt_event_count",
    "pred_event_count",
    "tp",
    "fn",
    "fp",
    "all_correct",
)
EVENT_ERROR_FIELDS = (
    "pair_id",
    "class_id",
    "gt_status",
    "pred_status",
    "error_type",
    "gt_before_box",
    "gt_after_box",
    "pred_before_box",
    "pred_after_box",
)


def analyze_errors(
    predictions_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    iou_threshold: float,
    manifest_loader: Callable[[Path], object],
    cases_per_group: int = DEFAULT_CASES_PER_GROUP,
) -> dict:
    """Write pair_results.csv, event_errors.csv, selected_cases.json and summary.json.

    `manifest_loader` is the runner's own `load_manifest`, injected so invalid
    manifests are rejected exactly as in evaluation. Pair inventories must match
    exactly; a disagreement names the offending pair_id. Nothing is published
    unless every stage, including the evaluator-consistency check, succeeded.
    """
    if isinstance(cases_per_group, bool) or not isinstance(cases_per_group, int):
        raise TypeError("cases_per_group must be an integer")
    if cases_per_group < 1:
        raise ValueError(f"cases_per_group must be >= 1, got {cases_per_group}")
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        raise TypeError("iou_threshold must be a number")
    if not isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError(
            f"iou_threshold must be finite and in (0, 1], got {iou_threshold!r}"
        )

    predictions_file = Path(predictions_path)
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {target}")

    manifest = manifest_loader(Path(manifest_path))
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest loader must return a mapping")
    ground_truth = manifest["ground_truth"]
    if not isinstance(ground_truth, Mapping):
        raise TypeError("manifest ground_truth must be a mapping keyed by pair_id")
    eligible = list(manifest["eligible_no_change_pair_ids"])

    content = predictions_file.read_bytes()
    predictions = json.loads(content.decode("utf-8"))
    if not isinstance(predictions, dict):
        raise TypeError("predictions file must be a JSON object keyed by pair_id")
    _require_same_pairs(predictions, ground_truth)

    evaluator = EventEvaluator(iou_threshold=float(iou_threshold))
    aggregate = evaluator.evaluate(predictions, ground_truth)

    pair_rows, event_rows, decomposed = _classify_pairs(
        evaluator, predictions, ground_truth, set(eligible)
    )
    _require_consistent(decomposed, aggregate)

    no_change = _no_change_section(predictions, eligible)
    summary = {
        "pair_count": len(ground_truth),
        "eligible_no_change_pair_count": len(eligible),
        "split": manifest["split"],
        "dataset_release": manifest["dataset_release"],
        "per_status": aggregate["per_status"],
        "error_type_counts": {
            error_type: sum(1 for row in event_rows if row["error_type"] == error_type)
            for error_type in ERROR_TYPES
        },
        "inputs": {
            "predictions_sha256": hashlib.sha256(content).hexdigest(),
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        },
        "rules": {
            "event_protocol_version": aggregate["protocol_version"],
            "event_iou_threshold": aggregate["iou_threshold"],
            "event_iou_threshold_source": "user-supplied iou_threshold",
            "event_zero_division": aggregate["zero_division"],
            "no_change_protocol_version": no_change["protocol_version"],
        },
        "other_reasons": [
            {"pair_id": row["pair_id"], "class_id": row["class_id"], "reason": row["reason"]}
            for row in event_rows if row["error_type"] == OTHER
        ],
        "verdict_source": "EventEvaluator/NoChangeEvaluator single-event verdicts",
    }
    files = {
        "pair_results.csv": _pair_csv(pair_rows),
        "event_errors.csv": _event_csv(event_rows),
        "selected_cases.json": _json_text(
            _selected_cases(event_rows, no_change, cases_per_group)
        ),
        "summary.json": _json_text(summary),
    }
    _publish(target, files)
    return {
        "pair_count": summary["pair_count"],
        "error_count": len(event_rows),
        "split": summary["split"],
        "dataset_release": summary["dataset_release"],
        "output_dir": str(target),
    }


def _require_same_pairs(predictions: Mapping, ground_truth: Mapping) -> None:
    missing = sorted(set(ground_truth) - set(predictions))
    extra = sorted(set(predictions) - set(ground_truth))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"manifest pairs missing from predictions: {missing}")
        if extra:
            parts.append(f"prediction pairs absent from manifest: {extra}")
        raise ValueError("; ".join(parts))


def _classify_pairs(
    evaluator: EventEvaluator,
    predictions: Mapping,
    ground_truth: Mapping,
    eligible: set[str],
) -> tuple[list[dict], list[dict], dict]:
    """Return pair rows, error rows and per-status counts from single-event verdicts."""
    decomposed = {status: {"tp": 0, "fp": 0, "fn": 0} for status in evaluator.STATUSES}
    pair_rows: list[dict] = []
    event_rows: list[dict] = []

    for pair_id in sorted(ground_truth):
        predicted = {event["class_id"]: event for event in predictions[pair_id]}
        truth = {event["class_id"]: event for event in ground_truth[pair_id]}
        tp = fn = fp = 0
        for class_id in sorted(set(predicted) | set(truth)):
            prediction = predicted.get(class_id)
            expected = truth.get(class_id)
            result, error_type, reason = _single_verdict(evaluator, prediction, expected)
            for status, counts in result["per_status"].items():
                for key in ("tp", "fp", "fn"):
                    decomposed[status][key] += counts[key]
            pair_counts = _sum_counts(result)
            tp += pair_counts["tp"]
            fn += pair_counts["fn"]
            fp += pair_counts["fp"]
            if error_type is not None:
                event_rows.append({
                    "pair_id": pair_id,
                    "class_id": class_id,
                    "gt_status": expected["status"] if expected is not None else "",
                    "pred_status": prediction["status"] if prediction is not None else "",
                    "error_type": error_type,
                    "reason": reason,
                    "gt_before_box": _box_cell(expected, "before_box"),
                    "gt_after_box": _box_cell(expected, "after_box"),
                    "pred_before_box": _box_cell(prediction, "before_box"),
                    "pred_after_box": _box_cell(prediction, "after_box"),
                })
        pair_rows.append({
            "pair_id": pair_id,
            "is_eligible_nochange": "true" if pair_id in eligible else "false",
            "gt_event_count": len(truth),
            "pred_event_count": len(predicted),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "all_correct": "true" if tp == len(truth) and fn == 0 and fp == 0 else "false",
        })
    return pair_rows, event_rows, decomposed


def _single_verdict(
    evaluator: EventEvaluator,
    prediction: dict | None,
    truth: dict | None,
) -> tuple[dict, str | None, str]:
    """Ask the evaluator about one class_id; classify its verdict, never re-score it."""
    if prediction is None:
        result = evaluator.evaluate({"pair": []}, {"pair": [truth]})
        counts = result["per_status"][truth["status"]]
        if counts["fn"] == 1 and counts["tp"] == 0 and counts["fp"] == 0:
            return result, MISSED_EVENT, "ground-truth event has no prediction for this class_id"
        return result, OTHER, _unexpected(result, "single ground-truth event")
    if truth is None:
        result = evaluator.evaluate({"pair": [prediction]}, {"pair": []})
        counts = result["per_status"][prediction["status"]]
        if counts["fp"] == 1 and counts["tp"] == 0 and counts["fn"] == 0:
            return result, SPURIOUS_EVENT, "prediction has no ground-truth event for this class_id"
        return result, OTHER, _unexpected(result, "single predicted event")

    result = evaluator.evaluate({"pair": [prediction]}, {"pair": [truth]})
    truth_counts = result["per_status"][truth["status"]]
    if truth_counts["tp"] == 1 and truth_counts["fp"] == 0 and truth_counts["fn"] == 0:
        return result, None, ""
    pred_counts = result["per_status"][prediction["status"]]
    paired = (truth_counts["fn"] == 1 and pred_counts["fp"] == 1
              and truth_counts["tp"] == 0 and pred_counts["tp"] == 0
              and sum(_sum_counts(result).values()) == 2)
    if not paired:
        return result, OTHER, _unexpected(result, "paired event")
    if prediction["status"] != truth["status"]:
        return result, WRONG_STATUS, (
            f"status {prediction['status']!r} disagrees with ground truth {truth['status']!r}"
        )
    return result, MISALIGNED_EVENT, (
        "status matches but the evaluator's localization check failed"
    )


def _sum_counts(result: Mapping) -> dict:
    total = {"tp": 0, "fp": 0, "fn": 0}
    for counts in result["per_status"].values():
        for key in total:
            total[key] += counts[key]
    return total


def _unexpected(result: Mapping, context: str) -> str:
    return (
        f"evaluator returned an unexpected verdict for a {context}: "
        f"{json.dumps(result['per_status'], sort_keys=True)}"
    )


def _require_consistent(decomposed: Mapping, aggregate: Mapping) -> None:
    differences = {
        status: {
            key: decomposed[status][key] - values[key]
            for key in ("tp", "fp", "fn")
            if decomposed[status][key] != values[key]
        }
        for status, values in aggregate["per_status"].items()
    }
    differences = {status: diff for status, diff in differences.items() if diff}
    if differences:
        raise RuntimeError(
            "single-event evaluator verdicts disagree with the aggregate evaluator "
            f"counts: {json.dumps(differences, sort_keys=True)}"
        )


def _no_change_section(predictions: Mapping, eligible: list[str]) -> dict:
    if not eligible:
        return {
            "evaluated": False,
            "protocol_version": None,
            "false_alarm_pair_ids": [],
            "success_pair_ids": [],
        }
    result = NoChangeEvaluator().evaluate(predictions, eligible)
    failures = set(result["false_alarm_pair_ids"])
    return {
        "evaluated": True,
        "protocol_version": result["protocol_version"],
        "false_alarm_pair_ids": sorted(failures),
        "success_pair_ids": [pid for pid in sorted(eligible) if pid not in failures],
    }


def _box_cell(event: dict | None, field: str) -> str:
    if event is None or event[field] is None:
        return ""
    return json.dumps(list(event[field]), separators=(",", ":"))


def _pair_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=PAIR_RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in PAIR_RESULT_FIELDS})
    return buffer.getvalue()


def _event_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EVENT_ERROR_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in EVENT_ERROR_FIELDS})
    return buffer.getvalue()


def _selected_cases(event_rows: list[dict], no_change: dict, limit: int) -> dict:
    """Deterministically keep the first `limit` rows of each (status, error_type) group."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in event_rows:  # already ordered by (pair_id, class_id)
        status = row["gt_status"] or row["pred_status"]
        grouped.setdefault((status, row["error_type"]), []).append(row)

    groups = []
    for (status, error_type) in sorted(grouped):
        rows = grouped[(status, error_type)]
        cases = [
            _case(row, status, error_type, rank, len(rows))
            for rank, row in enumerate(rows[:limit], start=1)
        ]
        groups.append({
            "status": status,
            "error_type": error_type,
            "total": len(rows),
            "cases": cases,
        })

    def pair_cases(pair_ids: list[str], error_type: str, reason: str) -> dict:
        return {
            "total": len(pair_ids),
            "cases": [
                {
                    "pair_id": pair_id,
                    "class_id": None,
                    "gt_status": None,
                    "pred_status": None,
                    "error_type": error_type,
                    "reason": f"{reason} (rank {rank} of {len(pair_ids)} by pair_id)",
                    "error_reason": None,
                }
                for rank, pair_id in enumerate(pair_ids[:limit], start=1)
            ],
        }

    return {
        "cases_per_group": limit,
        "ordering": "sorted by (pair_id, class_id); first N of each group kept",
        "groups": groups,
        "no_change": {
            "evaluated": no_change["evaluated"],
            "successes": pair_cases(
                no_change["success_pair_ids"],
                "none",
                "eligible no-change pair; the evaluator found no false alarm",
            ),
            "failures": pair_cases(
                no_change["false_alarm_pair_ids"],
                "false_alarm",
                "eligible no-change pair; the evaluator counted a change-status prediction",
            ),
        },
    }


def _case(row: dict, status: str, error_type: str, rank: int, total: int) -> dict:
    return {
        "pair_id": row["pair_id"],
        "class_id": row["class_id"],
        "gt_status": row["gt_status"] or None,
        "pred_status": row["pred_status"] or None,
        "error_type": error_type,
        "reason": (
            f"(pair_id, class_id) order rank {rank} of {total} in group "
            f"{status}/{error_type}"
        ),
        "error_reason": row["reason"] or None,
    }


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _publish(target: Path, files: dict[str, str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for name, text in files.items():
            (staging / name).write_text(text, encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {target}")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
