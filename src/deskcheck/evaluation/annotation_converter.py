"""Owner A: convert annotated pairs to the evaluator-facing event format.

The annotation schema (`data/annotations/pairs.jsonl`) uses object references
(`before_object_id`, `after_object_id`). B's `EventEvaluator` /
`NoChangeEvaluator` expect resolved boxes:

    {
      "pair_id": "p0001",
      "class_id": 0,
      "status": "moved",
      "before_box": [x1, y1, x2, y2] | None,
      "after_box": [x1, y1, x2, y2] | None,
    }

This module is the single, tested boundary between the two.

Rules:
- Every event resolves through its object references to actual boxes.
- Absent-side boxes are `None` (matching `added` / `removed`).
- Empty scenes produce an empty event list for that pair (valid).
- Invalid no-change eligibility (a change event inside a no-change pair)
  is rejected — the converter refuses to emit mismatched events silently.
"""

from __future__ import annotations

from typing import Any

from deskcheck.data.dataset_catalog import AnnotatedPair


def _box_for(objects: tuple[dict[str, Any], ...], object_id: str | None) -> list[float] | None:
    if object_id is None:
        return None
    for obj in objects:
        if obj["object_id"] == object_id:
            return list(obj["bbox_xyxy"])
    raise ValueError(f"object_id {object_id!r} not found in annotation")


def convert_pair(pair: AnnotatedPair) -> list[dict[str, Any]]:
    """Return evaluator-facing events for one pair.

    Raises ValueError if a change event appears in a no-change pair.
    """
    if pair.independently_captured_no_change:
        change_statuses = {"added", "removed", "moved"}
        offenders = [ev for ev in pair.events if ev["status"] in change_statuses]
        if offenders:
            classes = sorted({ev["class_id"] for ev in offenders})
            raise ValueError(
                f"{pair.pair_id}: independent no-change pair contains change "
                f"events for classes {classes}"
            )

    out: list[dict[str, Any]] = []
    for ev in pair.events:
        before_box = _box_for(pair.manual_before_objects, ev.get("before_object_id"))
        after_box = _box_for(pair.manual_after_objects, ev.get("after_object_id"))
        out.append({
            "pair_id": pair.pair_id,
            "class_id": ev["class_id"],
            "status": ev["status"],
            "before_box": before_box,
            "after_box": after_box,
        })
    return out


def convert_dataset(pairs: dict[str, AnnotatedPair]) -> dict[str, list[dict[str, Any]]]:
    """Convert every pair. Returns {pair_id: [events]}."""
    return {pid: convert_pair(p) for pid, p in pairs.items()}


def ground_truth_events(
    pairs: dict[str, AnnotatedPair],
    pair_ids: list[str],
) -> list[dict[str, Any]]:
    """Flat list of events restricted to the given pair_ids.

    This is the function B's EvaluationRunner should call when it needs
    ground truth for a split.
    """
    events: list[dict[str, Any]] = []
    for pid in pair_ids:
        if pid not in pairs:
            raise ValueError(f"pair_id {pid!r} not in dataset")
        events.extend(convert_pair(pairs[pid]))
    return events
