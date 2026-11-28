"""Read-only dataset split report built from annotation metadata only.

Answers three checklist items for one released dataset: pair / capture group /
object / status integrity, split-by-capture-group leakage together with status
coverage in val and test, and per-split counts.

What is reused, and from where
------------------------------
`dataset_catalog.load_catalog` is the only loader: one parser, one schema rule
set. It is strict, so a structural violation (duplicate `pair_id`, a `pair_id`
in two splits, a split citing an unknown `pair_id`, a record missing a required
field) raises before a report exists; the CLI relays that message untouched and
writes nothing. `eval_manifest`'s public `CHANGE_STATUSES` and `REQUIRED_SPLITS`
are reused verbatim, as are the public `convert_pair` converter and
`DEFAULT_CLASS_NAMES` that `eval_manifest` validates with.

What is written here, and why
-----------------------------
`eval_manifest` exposes its split-level and per-pair rules only through private
helpers (`_validate_pair_metadata`, `_convert_named_pair`, `_BOX_PRESENCE`), and
those helpers stop at the first offender instead of reporting every offender.
This module therefore implements, from the loaded catalog and `splits.json`:
pair assignment counts, release comparison, capture group leakage, status
coverage, class-id range, status/box presence and object-count reconciliation.
Per-pair conversion and the no-change-pair rule go through the public
`convert_pair`, so there is no second event rule set.

Annotation metadata only
------------------------
No image is opened, decoded, stat-ed or hashed, and no evaluation result,
prediction or model output is read. Image paths are carried through as strings.

Capture groups are never inferred from pair IDs. If a record's
`capture_group_id` is not a non-empty string the leakage check reports
`cannot_be_verified` and names that field instead of guessing.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from deskcheck.data.dataset_catalog import (
    VALID_STATUSES,
    AnnotatedPair,
    SplitManifest,
    load_catalog,
)
from deskcheck.data.yolo_exporter import DEFAULT_CLASS_NAMES
from deskcheck.evaluation.annotation_converter import convert_pair
from deskcheck.evaluation.eval_manifest import CHANGE_STATUSES, REQUIRED_SPLITS

REPORT_SCHEMA_VERSION = "1.0"
REPORT_KIND = "dataset-split-report"
TOTAL_KEY = "total"
SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")
STATUS_ORDER: tuple[str, ...] = tuple(sorted(VALID_STATUSES))

# (before_present, after_present) per status. eval_manifest holds the identical
# table privately; it is restated here rather than imported.
STATUS_BOX_PRESENCE: dict[str, tuple[bool, bool]] = {
    "added": (False, True),
    "removed": (True, False),
    "moved": (True, True),
    "unchanged": (True, True),
}

METADATA_ONLY_STATEMENT = (
    "Built from annotation metadata only: no image file was opened, decoded, stat-ed "
    "or hashed, no evaluation result, prediction or model output was read, and no "
    "network access was used. The report is therefore safe to produce for the test "
    "split without exposing test images."
)


def build_dataset_report(
    pairs_path: str | Path,
    splits_path: str | Path,
    dataset_release: str,
) -> dict[str, Any]:
    """Return a JSON-serializable split report for one released dataset.

    Rejects an unnamed release, a release that differs from `splits.json`, and
    an empty pairs file. Structural violations are rejected by the strict
    loader. Reported data problems never raise; they are recorded as failing or
    unverifiable checks so the whole picture is visible in one run.
    """
    release = _checked_release(dataset_release)
    source = Path(pairs_path)
    splits_source = Path(splits_path)
    pairs, manifest = load_catalog(source, splits_source)
    if not pairs:
        raise ValueError(f"pairs file has no pairs: {source}")
    if release != manifest.dataset_release:
        raise ValueError(
            f"dataset release mismatch: {release!r} does not match "
            f"splits.json dataset_release {manifest.dataset_release!r}"
        )

    split_pairs = {
        name: [pairs[pair_id] for pair_id in manifest.splits[name] if pair_id in pairs]
        for name in SPLIT_ORDER
    }
    split_stats = {name: _stats_for(split_pairs[name]) for name in SPLIT_ORDER}
    assigned = [pair for name in SPLIT_ORDER for pair in split_pairs[name]]
    split_stats[TOTAL_KEY] = _stats_for(assigned)
    checks = _build_checks(pairs, manifest, split_pairs, release)

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "dataset_release": release,
        "inputs": {
            "pairs": _file_identity(source),
            "splits": _file_identity(splits_source),
        },
        "splits_manifest": {
            "schema_version": manifest.schema_version,
            "seed": manifest.seed,
            "dataset_release": manifest.dataset_release,
        },
        "annotation_metadata_only": True,
        "metadata_only_statement": METADATA_ONLY_STATEMENT,
        "splits": split_stats,
        "dataset": _stats_for(pairs.values()),
        "checks": checks,
        "summary": _summary(checks),
    }


def report_to_json(report: Mapping[str, Any]) -> str:
    """Serialize deterministically: sorted keys, fixed indent, trailing newline."""
    return json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"


def render_markdown(
    report: Mapping[str, Any],
    *,
    expect_pairs: Sequence[int] | None = None,
    expect_objects: int | None = None,
) -> str:
    """Render the report as tables. Expectations are informational only."""
    lines: list[str] = []
    add = lines.append
    add("# Dataset split report")
    add("")
    add(f"- Dataset release: `{report['dataset_release']}`")
    add(f"- Report schema version: `{report['report_schema_version']}`")
    for role, label in (("pairs", "pairs"), ("splits", "splits")):
        identity = report["inputs"][role]
        add(
            f"- {label} file: `{identity['name']}` sha256 `{identity['sha256']}` "
            f"({identity['size_bytes']} bytes)"
        )
    manifest = report["splits_manifest"]
    add(
        f"- splits.json: release `{manifest['dataset_release']}`, seed "
        f"{manifest['seed']}, schema version `{manifest['schema_version']}`"
    )
    add("")
    add("## Metadata only")
    add("")
    add(f"> {report['metadata_only_statement']}")
    add("")

    add("## Counts per split")
    add("")
    rows = [
        (
            name,
            *_count_cells(report["splits"][name]),
        )
        for name in (*SPLIT_ORDER, TOTAL_KEY)
    ]
    lines.extend(_table(
        (
            "split",
            "pairs",
            "images",
            "capture groups",
            "objects",
            "no-change pairs",
            "eligible no-change pairs",
        ),
        rows,
    ))
    add("")
    add(
        "`total` covers the union of the three splits, so its image and capture group "
        "counts are distinct counts, not sums of the rows above."
    )
    add("")

    class_names = list(DEFAULT_CLASS_NAMES)
    add("## Objects per class")
    add("")
    class_rows: list[tuple[Any, ...]] = [
        (name, *(report["splits"][split]["objects_by_class"][name]
                 for split in (*SPLIT_ORDER, TOTAL_KEY)))
        for name in class_names
    ]
    if any(report["splits"][name]["other_class_object_count"]
           for name in (*SPLIT_ORDER, TOTAL_KEY)):
        class_rows.append((
            "(other class ids)",
            *(report["splits"][split]["other_class_object_count"]
              for split in (*SPLIT_ORDER, TOTAL_KEY)),
        ))
    lines.extend(_table(("class", *SPLIT_ORDER, TOTAL_KEY), class_rows))
    add("")

    add("## Events per status")
    add("")
    lines.extend(_table(
        ("status", *SPLIT_ORDER, TOTAL_KEY),
        [
            (status, *(report["splits"][split]["events_by_status"][status]
                       for split in (*SPLIT_ORDER, TOTAL_KEY)))
            for status in STATUS_ORDER
        ],
    ))
    add("")

    add("## Pairs containing each status")
    add("")
    lines.extend(_table(
        ("status", *SPLIT_ORDER, TOTAL_KEY),
        [
            (status, *(report["splits"][split]["pairs_by_status"][status]
                       for split in (*SPLIT_ORDER, TOTAL_KEY)))
            for status in STATUS_ORDER
        ],
    ))
    add("")

    add("## Integrity checks")
    add("")
    lines.extend(_left_aligned_table(
        ("#", "check", "status", "detail"),
        [
            (index, check["id"], check["status"], check["detail"])
            for index, check in enumerate(report["checks"], start=1)
        ],
    ))
    add("")

    add("## Offending entries")
    add("")
    flagged = [check for check in report["checks"] if check["status"] != "pass"]
    if not flagged:
        add("None; every check passed.")
        add("")
    for check in flagged:
        add(f"### `{check['id']}` - {check['status']}")
        add("")
        if not check["offending"]:
            add(f"- no individual entries; detail: {check['detail']}")
        for item in check["offending"]:
            add(f"- `{_render_value(item)}`")
        add("")

    expectation_rows = _expectation_rows(report, expect_pairs, expect_objects)
    if expectation_rows:
        add("## Differences from expected")
        add("")
        add(
            "> Informational only. These expected numbers are not part of the integrity "
            "checks above and never change a check result, `report.json` or the exit code."
        )
        add("")
        lines.extend(_table(
            ("item", "expected", "reported", "result"), expectation_rows,
        ))
        add("")

    return "\n".join(lines).rstrip("\n") + "\n"


def _checked_release(dataset_release: str) -> str:
    if not isinstance(dataset_release, str) or not dataset_release.strip():
        raise ValueError("dataset release must be a nonempty string")
    return dataset_release.strip()


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"name": path.name, "sha256": digest.hexdigest(), "size_bytes": size}


def _usable_group_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _empty_stats() -> dict[str, Any]:
    return {
        "pair_count": 0,
        "image_count": 0,
        "capture_group_count": 0,
        "object_count": 0,
        "objects_by_class": {name: 0 for name in DEFAULT_CLASS_NAMES},
        "other_class_object_count": 0,
        "events_by_status": {status: 0 for status in STATUS_ORDER},
        "pairs_by_status": {status: 0 for status in STATUS_ORDER},
        "no_change_pairs": 0,
        "eligible_no_change_pairs": 0,
    }


def _stats_for(pairs: Iterable[AnnotatedPair]) -> dict[str, Any]:
    stats = _empty_stats()
    images: set[str] = set()
    groups: set[str] = set()
    for pair in pairs:
        stats["pair_count"] += 1
        images.add(pair.before_image_id)
        images.add(pair.after_image_id)
        if _usable_group_id(pair.capture_group_id):
            groups.add(pair.capture_group_id)
        for obj in (*pair.manual_before_objects, *pair.manual_after_objects):
            stats["object_count"] += 1
            class_id = obj.get("class_id")
            if type(class_id) is int and 0 <= class_id < len(DEFAULT_CLASS_NAMES):
                stats["objects_by_class"][DEFAULT_CLASS_NAMES[class_id]] += 1
            else:
                stats["other_class_object_count"] += 1
        pair_statuses = {event.get("status") for event in pair.events}
        for status in pair_statuses:
            if status in stats["pairs_by_status"]:
                stats["pairs_by_status"][status] += 1
        for event in pair.events:
            status = event.get("status")
            if status in stats["events_by_status"]:
                stats["events_by_status"][status] += 1
        if not pair_statuses & CHANGE_STATUSES:
            stats["no_change_pairs"] += 1
        if pair.independently_captured_no_change:
            stats["eligible_no_change_pairs"] += 1
    stats["image_count"] = len(images)
    stats["capture_group_count"] = len(groups)
    return stats


def _result(
    check_id: str,
    title: str,
    status: str,
    detail: str,
    offending: Sequence[Any] = (),
) -> dict[str, Any]:
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "detail": detail,
        "offending": list(offending),
    }


def _build_checks(
    pairs: Mapping[str, AnnotatedPair],
    manifest: SplitManifest,
    split_pairs: Mapping[str, Sequence[AnnotatedPair]],
    release: str,
) -> list[dict[str, Any]]:
    manifest_entries = [
        (name, pair_id) for name in SPLIT_ORDER for pair_id in manifest.splits[name]
    ]
    checks = [
        _manifest_ids_exist(pairs, manifest_entries),
        _pairs_file_ids_assigned(pairs, manifest_entries),
        _pair_ids_unique(manifest),
        _result(
            "dataset_release_matches",
            "--dataset-release equals the release recorded in splits.json",
            "pass",
            f"{release!r} matches splits.json dataset_release {manifest.dataset_release!r}",
        ),
        _capture_group_leakage(pairs, split_pairs),
    ]
    checks.extend(_status_coverage(split_pairs))
    class_offenders = _class_id_offenders(pairs)
    checks.append(_result(
        "class_ids_known",
        f"Every object and event class_id is one of the {len(DEFAULT_CLASS_NAMES)} known classes",
        "fail" if class_offenders else "pass",
        (
            f"{len(class_offenders)} class_id value(s) outside 0..{len(DEFAULT_CLASS_NAMES) - 1}"
            if class_offenders
            else f"all class_id values are within 0..{len(DEFAULT_CLASS_NAMES) - 1}"
        ),
        class_offenders,
    ))
    conversion_offenders, presence_offenders = _conversion_and_presence(pairs)
    checks.append(_result(
        "annotation_converter_accepts_pairs",
        "The public annotation converter accepts every pair",
        "fail" if conversion_offenders else "pass",
        (
            f"{len(conversion_offenders)} pair(s) rejected by convert_pair"
            if conversion_offenders
            else f"convert_pair accepted all {len(pairs)} pair(s)"
        ),
        conversion_offenders,
    ))
    checks.append(_result(
        "status_box_presence_agrees",
        "Event status agrees with before/after box presence",
        "fail" if presence_offenders else "pass",
        (
            f"{len(presence_offenders)} event(s) disagree with their status"
            if presence_offenders
            else "every event's boxes match the presence required by its status"
        ),
        presence_offenders,
    ))
    checks.append(_object_counts_reconcile(pairs, split_pairs))
    return checks


def _manifest_ids_exist(
    pairs: Mapping[str, AnnotatedPair],
    manifest_entries: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    unknown = sorted({pair_id for _, pair_id in manifest_entries if pair_id not in pairs})
    return _result(
        "manifest_pair_ids_exist",
        "Every pair_id in splits.json exists in the pairs file",
        "fail" if unknown else "pass",
        (
            f"{len(unknown)} pair_id(s) listed in splits.json are absent from the pairs file"
            if unknown
            else f"all {len(manifest_entries)} split entries resolve to a pair"
        ),
        unknown,
    )


def _pairs_file_ids_assigned(
    pairs: Mapping[str, AnnotatedPair],
    manifest_entries: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    unassigned = sorted(set(pairs) - {pair_id for _, pair_id in manifest_entries})
    return _result(
        "pairs_file_pair_ids_assigned",
        "Every pair in the pairs file appears in a split",
        "fail" if unassigned else "pass",
        (
            f"{len(unassigned)} pair(s) in the pairs file appear in no split"
            if unassigned
            else f"all {len(pairs)} pair(s) are assigned to a split"
        ),
        unassigned,
    )


def _pair_ids_unique(manifest: SplitManifest) -> dict[str, Any]:
    counts = Counter(
        pair_id for name in SPLIT_ORDER for pair_id in manifest.splits[name]
    )
    duplicated = sorted(pair_id for pair_id, count in counts.items() if count > 1)
    offending = [
        {
            "pair_id": pair_id,
            "splits": [name for name in SPLIT_ORDER if pair_id in manifest.splits[name]],
        }
        for pair_id in duplicated
    ]
    return _result(
        "pair_ids_unique",
        "No pair_id appears in more than one split",
        "fail" if duplicated else "pass",
        (
            f"{len(duplicated)} pair_id(s) appear in more than one split"
            if duplicated
            else "every pair_id appears at most once across train, val and test"
        ),
        offending,
    )


def _capture_group_leakage(
    pairs: Mapping[str, AnnotatedPair],
    split_pairs: Mapping[str, Sequence[AnnotatedPair]],
) -> dict[str, Any]:
    unusable = sorted({
        pair.pair_id
        for name in SPLIT_ORDER
        for pair in split_pairs[name]
        if not _usable_group_id(pair.capture_group_id)
    })
    title = "No capture group appears in more than one split"
    if unusable:
        return _result(
            "capture_group_leakage",
            title,
            "cannot_be_verified",
            "the pairs file exposes no usable 'capture_group_id' field for "
            f"{len(unusable)} pair(s); capture groups are never inferred from pair_ids",
            unusable,
        )
    group_splits: dict[str, list[str]] = {}
    for name in SPLIT_ORDER:
        for pair in split_pairs[name]:
            names = group_splits.setdefault(pair.capture_group_id, [])
            if name not in names:
                names.append(name)
    leaking = sorted(group for group, names in group_splits.items() if len(names) > 1)
    offending = [
        {"capture_group_id": group, "splits": group_splits[group]} for group in leaking
    ]
    return _result(
        "capture_group_leakage",
        title,
        "fail" if leaking else "pass",
        (
            f"{len(leaking)} of {len(group_splits)} capture group(s) span more than one split"
            if leaking
            else f"all {len(group_splits)} capture group(s) stay within a single split"
        ),
        offending,
    )


def _status_coverage(
    split_pairs: Mapping[str, Sequence[AnnotatedPair]],
) -> list[dict[str, Any]]:
    checks = []
    for split in REQUIRED_SPLITS:
        present = {
            event.get("status")
            for pair in split_pairs[split]
            for event in pair.events
        }
        missing = [status for status in STATUS_ORDER if status not in present]
        checks.append(_result(
            f"status_coverage_{split}",
            f"Every status appears at least once in {split}",
            "fail" if missing else "pass",
            (
                f"missing from {split}: {missing}"
                if missing
                else f"{split} covers all {len(STATUS_ORDER)} statuses"
            ),
            missing,
        ))
    return checks


def _class_id_offenders(pairs: Mapping[str, AnnotatedPair]) -> list[dict[str, Any]]:
    offenders = []
    known = range(len(DEFAULT_CLASS_NAMES))
    for pair_id in sorted(pairs):
        pair = pairs[pair_id]
        for side in ("manual_before_objects", "manual_after_objects"):
            for obj in getattr(pair, side):
                class_id = obj.get("class_id")
                if type(class_id) is not int or class_id not in known:
                    offenders.append({
                        "pair_id": pair_id,
                        "side": side,
                        "object_id": obj.get("object_id"),
                        "class_id": class_id,
                    })
    return offenders


def _conversion_and_presence(
    pairs: Mapping[str, AnnotatedPair],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversion_offenders: list[dict[str, Any]] = []
    presence_offenders: list[dict[str, Any]] = []
    for pair_id in sorted(pairs):
        try:
            events = convert_pair(pairs[pair_id])
        except ValueError as exc:
            conversion_offenders.append({"pair_id": pair_id, "error": str(exc)})
            continue
        for event in events:
            expected = STATUS_BOX_PRESENCE.get(event["status"])
            actual = (event["before_box"] is not None, event["after_box"] is not None)
            if expected != actual:
                presence_offenders.append({
                    "pair_id": pair_id,
                    "class_id": event["class_id"],
                    "status": event["status"],
                    "before_box_present": actual[0],
                    "after_box_present": actual[1],
                })
    return conversion_offenders, presence_offenders


def _object_count(pairs: Iterable[AnnotatedPair]) -> int:
    return sum(
        len(pair.manual_before_objects) + len(pair.manual_after_objects)
        for pair in pairs
    )


def _object_counts_reconcile(
    pairs: Mapping[str, AnnotatedPair],
    split_pairs: Mapping[str, Sequence[AnnotatedPair]],
) -> dict[str, Any]:
    across = _object_count(
        pair for name in SPLIT_ORDER for pair in split_pairs[name]
    )
    in_file = _object_count(pairs.values())
    offending = [] if across == in_file else [{"across_splits": across, "pairs_file": in_file}]
    return _result(
        "object_counts_reconcile",
        "Total objects across splits equals the total in the pairs file",
        "fail" if offending else "pass",
        f"objects across splits: {across}; objects in the pairs file: {in_file}",
        offending,
    )


def _summary(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(check["status"] for check in checks)
    failed = counts["fail"]
    unverified = counts["cannot_be_verified"]
    return {
        "check_count": len(checks),
        "passed": counts["pass"],
        "failed": failed,
        "unverified": unverified,
        "ok": failed == 0 and unverified == 0,
    }


def _count_cells(stats: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        stats["pair_count"],
        stats["image_count"],
        stats["capture_group_count"],
        stats["object_count"],
        stats["no_change_pairs"],
        stats["eligible_no_change_pairs"],
    )


def _render_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={value[key]}" for key in sorted(value))
    return str(value)


def _table(
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _left_aligned_table(
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _expectation_rows(
    report: Mapping[str, Any],
    expect_pairs: Sequence[int] | None,
    expect_objects: int | None,
) -> list[tuple[str, int, int, str]]:
    rows: list[tuple[str, int, int, str]] = []
    if expect_pairs is not None:
        for name, expected in zip(SPLIT_ORDER, expect_pairs, strict=True):
            reported = report["splits"][name]["pair_count"]
            rows.append((
                f"pairs ({name})",
                expected,
                reported,
                "match" if expected == reported else "differs",
            ))
    if expect_objects is not None:
        reported = report["dataset"]["object_count"]
        rows.append((
            "objects (pairs file)",
            expect_objects,
            reported,
            "match" if expect_objects == reported else "differs",
        ))
    return rows
