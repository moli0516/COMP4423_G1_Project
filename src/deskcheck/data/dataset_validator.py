"""Owner A: audit the dataset against Part A acceptance criteria.

Pure logic over loaded records. Does not open images or call the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest
from deskcheck.domain.bounding_box import BoundingBox

VALID_STATUSES = ("added", "removed", "moved", "unchanged")
REQUIRED_CLASS_IDS = frozenset(range(6))


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    pair_counts: dict[str, int] = field(default_factory=dict)
    unique_image_counts: dict[str, int] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)
    class_coverage: dict[str, list[int]] = field(default_factory=dict)
    status_coverage: dict[str, list[str]] = field(default_factory=dict)
    independent_no_change_counts: dict[str, int] = field(default_factory=dict)


# ---------- per-pair checks ----------


def _check_boxes(pair: AnnotatedPair, errors: list[str]) -> None:
    for side, objs in (("before", pair.manual_before_objects),
                       ("after", pair.manual_after_objects)):
        for obj in objs:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            try:
                BoundingBox(x1, y1, x2, y2).validate(pair.width, pair.height)
            except ValueError as exc:
                errors.append(f"{pair.pair_id} {side} {obj['object_id']}: {exc}")


def _check_class_ids(pair: AnnotatedPair, errors: list[str],
                     required_class_ids: frozenset[int]) -> None:
    for side, objs in (("before", pair.manual_before_objects),
                       ("after", pair.manual_after_objects)):
        for obj in objs:
            if obj["class_id"] not in required_class_ids:
                errors.append(
                    f"{pair.pair_id} {side} {obj['object_id']}: class_id "
                    f"{obj['class_id']} not in {sorted(required_class_ids)}"
                )


def _check_unique_object_ids(pair: AnnotatedPair, errors: list[str]) -> None:
    for side, objs in (("before", pair.manual_before_objects),
                       ("after", pair.manual_after_objects)):
        ids = [o["object_id"] for o in objs]
        if len(ids) != len(set(ids)):
            errors.append(f"{pair.pair_id} {side}: duplicate object_id in {ids}")


def _check_single_instance_per_class(pair: AnnotatedPair, errors: list[str]) -> None:
    for side, objs in (("before", pair.manual_before_objects),
                       ("after", pair.manual_after_objects)):
        class_ids = [o["class_id"] for o in objs]
        if len(class_ids) != len(set(class_ids)):
            errors.append(
                f"{pair.pair_id} {side}: duplicate class_id in {sorted(class_ids)}"
            )


def _check_correspondences(pair: AnnotatedPair, errors: list[str]) -> None:
    """Correspondences must be same-class, one-to-one, and only between
    objects that exist on both sides."""
    before_by_id = {o["object_id"]: o for o in pair.manual_before_objects}
    after_by_id = {o["object_id"]: o for o in pair.manual_after_objects}

    matched_before: set[str] = set()
    matched_after: set[str] = set()
    for corr in pair.correspondences:
        bid = corr.get("before_object_id")
        aid = corr.get("after_object_id")
        cid = corr.get("class_id")

        if bid not in before_by_id:
            errors.append(f"{pair.pair_id}: correspondence references unknown before "
                          f"{bid!r}")
            continue
        if aid not in after_by_id:
            errors.append(f"{pair.pair_id}: correspondence references unknown after "
                          f"{aid!r}")
            continue

        b_obj = before_by_id[bid]
        a_obj = after_by_id[aid]
        if b_obj["class_id"] != a_obj["class_id"]:
            errors.append(f"{pair.pair_id}: correspondence {bid!r}/{aid!r} mixes "
                          f"classes {b_obj['class_id']} vs {a_obj['class_id']}")
        if cid is not None and cid != b_obj["class_id"]:
            errors.append(f"{pair.pair_id}: correspondence class_id {cid} disagrees "
                          f"with objects ({b_obj['class_id']})")

        if bid in matched_before:
            errors.append(f"{pair.pair_id}: before {bid!r} in multiple correspondences")
        if aid in matched_after:
            errors.append(f"{pair.pair_id}: after {aid!r} in multiple correspondences")
        matched_before.add(bid)
        matched_after.add(aid)


def _check_events(pair: AnnotatedPair, errors: list[str]) -> None:
    """Events must resolve to real objects, agree with side presence, cover
    every class that appears on either side, and have no absent-both rows."""
    before_by_class = {o["class_id"]: o for o in pair.manual_before_objects}
    after_by_class = {o["class_id"]: o for o in pair.manual_after_objects}
    before_by_id = {o["object_id"]: o for o in pair.manual_before_objects}
    after_by_id = {o["object_id"]: o for o in pair.manual_after_objects}

    union_classes = set(before_by_class) | set(after_by_class)
    seen_classes: set[int] = set()

    for ev in pair.events:
        cid = ev.get("class_id")
        status = ev.get("status")
        bid = ev.get("before_object_id")
        aid = ev.get("after_object_id")

        if status not in VALID_STATUSES:
            errors.append(f"{pair.pair_id}: event has invalid status {status!r}")
            continue
        if cid is None:
            errors.append(f"{pair.pair_id}: event missing class_id")
            continue
        if cid in seen_classes:
            errors.append(f"{pair.pair_id}: duplicate event for class_id {cid}")
        seen_classes.add(cid)

        in_before = cid in before_by_class
        in_after = cid in after_by_class
        if not in_before and not in_after:
            errors.append(f"{pair.pair_id}: event for class {cid} absent from both "
                          f"sides (must produce no row)")
        if status == "added":
            if not (not in_before and in_after):
                errors.append(f"{pair.pair_id}: 'added' for class {cid} requires "
                              f"absent-before + present-after")
            if bid is not None:
                errors.append(f"{pair.pair_id}: 'added' class {cid} must have "
                              f"before_object_id=null")
            if aid is None or aid not in after_by_id:
                errors.append(f"{pair.pair_id}: 'added' class {cid} needs a valid "
                              f"after_object_id")
        elif status == "removed":
            if not (in_before and not in_after):
                errors.append(f"{pair.pair_id}: 'removed' for class {cid} requires "
                              f"present-before + absent-after")
            if aid is not None:
                errors.append(f"{pair.pair_id}: 'removed' class {cid} must have "
                              f"after_object_id=null")
            if bid is None or bid not in before_by_id:
                errors.append(f"{pair.pair_id}: 'removed' class {cid} needs a valid "
                              f"before_object_id")
        elif status in ("moved", "unchanged"):
            if not (in_before and in_after):
                errors.append(f"{pair.pair_id}: '{status}' for class {cid} requires "
                              f"both sides present")
            if bid is None or bid not in before_by_id:
                errors.append(f"{pair.pair_id}: '{status}' class {cid} needs a valid "
                              f"before_object_id")
            if aid is None or aid not in after_by_id:
                errors.append(f"{pair.pair_id}: '{status}' class {cid} needs a valid "
                              f"after_object_id")

    # Every class appearing on either side must have exactly one event.
    missing = union_classes - seen_classes
    if missing:
        errors.append(f"{pair.pair_id}: classes {sorted(missing)} present in image "
                      f"but no event")


def _check_no_change_purity(pair: AnnotatedPair, errors: list[str]) -> None:
    """Independent no-change pairs must have zero change events."""
    if not pair.independently_captured_no_change:
        return
    change_statuses = {"added", "removed", "moved"}
    offenders = [ev for ev in pair.events if ev["status"] in change_statuses]
    if offenders:
        classes = sorted({ev["class_id"] for ev in offenders})
        errors.append(
            f"{pair.pair_id}: independent no-change pair contains change events "
            f"for classes {classes}"
        )


def _check_movement_consistency(
    pair: AnnotatedPair, errors: list[str], tau: float
) -> None:
    """Optional: if a frozen tau is supplied in the manifest, verify
    moved/unchanged statuses match the documented distance rule."""
    before_by_id = {o["object_id"]: o for o in pair.manual_before_objects}
    after_by_id = {o["object_id"]: o for o in pair.manual_after_objects}
    diag = hypot(pair.width, pair.height)
    for ev in pair.events:
        if ev["status"] not in ("moved", "unchanged"):
            continue
        bid = ev["before_object_id"]
        aid = ev["after_object_id"]
        if bid not in before_by_id or aid not in after_by_id:
            continue  # already flagged elsewhere
        b = before_by_id[bid]["bbox_xyxy"]
        a = after_by_id[aid]["bbox_xyxy"]
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        d = hypot(acx - bcx, acy - bcy) / diag
        expected = "moved" if d > tau else "unchanged"
        if ev["status"] != expected:
            errors.append(
                f"{pair.pair_id}: class {ev['class_id']} status {ev['status']!r} "
                f"inconsistent with d={d:.4f} (tau={tau} -> {expected})"
            )


# ---------- cross-pair checks ----------


def _check_group_integrity(
    pairs: dict[str, AnnotatedPair], manifest: SplitManifest, errors: list[str]
) -> None:
    group_to_split: dict[str, str] = {}
    image_to_split: dict[str, str] = {}
    for split_name, pair_ids in manifest.splits.items():
        for pair_id in pair_ids:
            pair = pairs[pair_id]
            gid = pair.capture_group_id
            if gid in group_to_split and group_to_split[gid] != split_name:
                errors.append(
                    f"capture_group_id {gid!r} crosses splits "
                    f"({group_to_split[gid]!r} and {split_name!r})"
                )
            group_to_split[gid] = split_name
            for image_id in (pair.before_image_id, pair.after_image_id):
                if image_id in image_to_split and image_to_split[image_id] != split_name:
                    errors.append(
                        f"image_id {image_id!r} shared across splits "
                        f"({image_to_split[image_id]!r} and {split_name!r})"
                    )
                image_to_split[image_id] = split_name


def _coverage(
    pairs: dict[str, AnnotatedPair], manifest: SplitManifest
) -> tuple[dict[str, list[int]], dict[str, list[str]], dict[str, int]]:
    class_cov: dict[str, list[int]] = {}
    status_cov: dict[str, list[str]] = {}
    nc_counts: dict[str, int] = {}
    for split_name, pair_ids in manifest.splits.items():
        class_ids: set[int] = set()
        statuses: set[str] = set()
        no_change = 0
        for pair_id in pair_ids:
            pair = pairs[pair_id]
            for obj in (*pair.manual_before_objects, *pair.manual_after_objects):
                class_ids.add(obj["class_id"])
            for ev in pair.events:
                statuses.add(ev["status"])
            if pair.independently_captured_no_change:
                no_change += 1
        class_cov[split_name] = sorted(class_ids)
        status_cov[split_name] = sorted(statuses)
        nc_counts[split_name] = no_change
    return class_cov, status_cov, nc_counts


# ---------- public API ----------


def audit(
    pairs: dict[str, AnnotatedPair],
    manifest: SplitManifest,
    required_class_ids: frozenset[int] = REQUIRED_CLASS_IDS,
    movement_threshold: float | None = None,
) -> AuditReport:
    """Run Part A checks.

    Returns a report; never raises for data problems.
    If `movement_threshold` is None, moved/unchanged consistency is not
    checked and a warning notes that limitation.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not pairs:
        errors.append("dataset is empty")
        return AuditReport(ok=False, errors=tuple(errors))

    manifest_ids = {pid for ids in manifest.splits.values() for pid in ids}
    unassigned = pairs.keys() - manifest_ids
    if unassigned:
        warnings.append(f"annotations not in manifest: {sorted(unassigned)}")

    for pair in pairs.values():
        _check_boxes(pair, errors)
        _check_class_ids(pair, errors, required_class_ids)
        _check_unique_object_ids(pair, errors)
        _check_single_instance_per_class(pair, errors)
        _check_correspondences(pair, errors)
        _check_events(pair, errors)
        _check_no_change_purity(pair, errors)
        if movement_threshold is not None:
            _check_movement_consistency(pair, errors, movement_threshold)

    if movement_threshold is None:
        warnings.append(
            "movement threshold not supplied; moved/unchanged consistency "
            "was not verified. Freeze tau before test annotation."
        )

    _check_group_integrity(pairs, manifest, errors)

    class_cov, status_cov, nc_counts = _coverage(pairs, manifest)

    for split_name, class_ids in class_cov.items():
        missing = required_class_ids - set(class_ids)
        if missing:
            errors.append(f"{split_name} split missing classes {sorted(missing)}")
    for split_name in ("val", "test"):
        missing_statuses = set(VALID_STATUSES) - set(status_cov.get(split_name, []))
        if missing_statuses:
            errors.append(
                f"{split_name} split missing statuses {sorted(missing_statuses)}"
            )
        if nc_counts.get(split_name, 0) == 0:
            errors.append(f"{split_name} split has zero independent no-change pairs")

    pair_counts = {name: len(ids) for name, ids in manifest.splits.items()}
    image_counts: dict[str, int] = {}
    for name, ids in manifest.splits.items():
        imgs: set[str] = set()
        for pid in ids:
            pair = pairs[pid]
            imgs.add(pair.before_image_id)
            imgs.add(pair.after_image_id)
        image_counts[name] = len(imgs)
    group_counts: dict[str, int] = {}
    for name, ids in manifest.splits.items():
        group_counts[name] = len({pairs[pid].capture_group_id for pid in ids})

    return AuditReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        pair_counts=pair_counts,
        unique_image_counts=image_counts,
        group_counts=group_counts,
        class_coverage=class_cov,
        status_coverage=status_cov,
        independent_no_change_counts=nc_counts,
    )
