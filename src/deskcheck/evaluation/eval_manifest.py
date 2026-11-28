"""Build the B evaluation manifest from released pair annotations.

Consumes only the pairs file metadata and its image path strings. Images are
never opened or decoded; referenced paths are only stat-ed for existence, the
same way the validation path already does (`load_manifest` resolves them
strictly). Image paths are written relative to the manifest file, never
absolute, so the manifest stays portable and byte-reproducible.

The payload is exactly what `scripts/evaluate_real_pairs.py:load_manifest`
accepts and is verified through that same function before publication.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from deskcheck.data.dataset_catalog import AnnotatedPair, load_annotations
from deskcheck.data.yolo_exporter import DEFAULT_CLASS_NAMES
from deskcheck.evaluation.annotation_converter import convert_pair
from deskcheck.evaluation.event_evaluator import EventEvaluator
from deskcheck.evaluation.no_change_evaluator import NoChangeEvaluator

MANIFEST_SCHEMA_VERSION = "1.0"
REQUIRED_SPLITS: tuple[str, ...] = ("val", "test")
CHANGE_STATUSES = frozenset({"added", "removed", "moved"})

# box presence per status: (before_present, after_present)
_BOX_PRESENCE = {
    "added": (False, True),
    "removed": (True, False),
    "moved": (True, True),
    "unchanged": (True, True),
}


def build_eval_manifest(
    pairs_path: str | Path,
    *,
    split: str,
    dataset_release: str,
    output: str | Path,
) -> dict:
    """Return a manifest payload for one split, with paths relative to `output`.

    Rejects an unknown split, an unnamed release, an empty pairs file, duplicate
    pair_ids, unknown classes, statuses that disagree with box presence, and
    eligible no-change pairs that carry change events. Every error names the
    offending pair.
    """
    if split not in REQUIRED_SPLITS:
        raise ValueError(f"split must be one of {list(REQUIRED_SPLITS)}, got {split!r}")
    if not isinstance(dataset_release, str) or not dataset_release.strip():
        raise ValueError("dataset release must be a nonempty string")

    source = Path(pairs_path)
    output_path = Path(output)
    output_dir = output_path.parent.resolve()
    pairs = load_annotations(source)
    if not pairs:
        raise ValueError(f"pairs file has no pairs: {source}")

    truth: dict[str, list[dict]] = {}
    for pair in pairs.values():
        _validate_pair_metadata(pair)
        truth[pair.pair_id] = _convert_named_pair(pair)

    # The runner's own contracts are the authority; check them before publishing.
    EventEvaluator().evaluate(truth, truth)
    eligible = [pid for pid, pair in pairs.items() if pair.independently_captured_no_change]
    if eligible:
        eligibility = NoChangeEvaluator().evaluate(truth, eligible)
        if eligibility["false_alarm_pair_count"]:
            raise ValueError(
                "Eligible no-change ground truth contains change events: "
                f"{eligibility['false_alarm_pair_ids']}"
            )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_release": dataset_release,
        "split": split,
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "before_path": _relative_image_path(
                    pair.pair_id, "before", pair.before_image_path, output_dir
                ),
                "after_path": _relative_image_path(
                    pair.pair_id, "after", pair.after_image_path, output_dir
                ),
            }
            for pair in pairs.values()
        ],
        "ground_truth": truth,
        "eligible_no_change_pair_ids": eligible,
    }


def publish_manifest(
    payload: dict,
    output: str | Path,
    *,
    verify: Callable[[Path], object],
) -> None:
    """Write `payload` to `output` only if `verify` accepts the staged file.

    The file is staged beside the destination and linked into place, so a
    rejected or crashed run never leaves a partial manifest at `output`.
    """
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        verify(staged)
        os.link(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _validate_pair_metadata(pair: AnnotatedPair) -> None:
    """Reject unknown classes and no-change pairs that carry change events."""
    known = range(len(DEFAULT_CLASS_NAMES))
    for objects, side in (
        (pair.manual_before_objects, "manual_before_objects"),
        (pair.manual_after_objects, "manual_after_objects"),
    ):
        for obj in objects:
            class_id = obj.get("class_id")
            if type(class_id) is not int or class_id not in known:
                raise ValueError(
                    f"pair {pair.pair_id!r}: unknown class_id {class_id!r} "
                    f"in {side} object {obj.get('object_id')!r}"
                )
    for event in pair.events:
        class_id = event.get("class_id")
        if type(class_id) is not int or class_id not in known:
            raise ValueError(
                f"pair {pair.pair_id!r}: unknown class_id {class_id!r} in event"
            )
    if pair.independently_captured_no_change:
        offenders = sorted(
            (event["status"], event["class_id"])
            for event in pair.events
            if event["status"] in CHANGE_STATUSES
        )
        if offenders:
            raise ValueError(
                f"pair {pair.pair_id!r}: eligible no-change pair has change events "
                f"(status, class_id): {offenders}"
            )


def _convert_named_pair(pair: AnnotatedPair) -> list[dict]:
    """Convert one pair, rejecting statuses that disagree with box presence."""
    try:
        events = convert_pair(pair)
    except ValueError as exc:
        raise ValueError(f"pair {pair.pair_id!r}: {exc}") from exc
    for event in events:
        expected = _BOX_PRESENCE[event["status"]]
        actual = (event["before_box"] is not None, event["after_box"] is not None)
        if actual != expected:
            raise ValueError(
                f"pair {pair.pair_id!r}: status {event['status']!r} for class "
                f"{event['class_id']} disagrees with box presence "
                f"(before={event['before_box'] is not None}, "
                f"after={event['after_box'] is not None})"
            )
    return events


def _relative_image_path(pair_id: str, side: str, image_path: str, output_dir: Path) -> str:
    path = Path(image_path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"pair {pair_id!r}: {side}_path does not exist: {path}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"pair {pair_id!r}: {side}_path is not a file: {path}")
    try:
        relative = os.path.relpath(resolved, output_dir)
    except ValueError as exc:
        raise ValueError(
            f"pair {pair_id!r}: {side}_path cannot be written relative to "
            f"{output_dir}: {resolved}"
        ) from exc
    return Path(relative).as_posix()
