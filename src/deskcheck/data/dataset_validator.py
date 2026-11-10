"""Owner A: audit the dataset against Part A acceptance criteria.

Pure logic over loaded records. Does not open images or call the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest
from deskcheck.domain.bounding_box import BoundingBox


VALID_STATUSES = ("added", "removed", "moved", "unchanged")
REQUIRED_CLASS_IDS = frozenset(range(6))  # 0..5 for the six approved classes


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


def _check_boxes(pair: AnnotatedPair, errors: list[str]) -> None:
    for side_name, objects in (
        ("before", pair.manual_before_objects),
        ("after", pair.manual_after_objects),
    ):
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            try:
                BoundingBox(x1, y1, x2, y2).validate(pair.width, pair.height)
            except ValueError as exc:
                errors.append(f"{pair.pair_id} {side_name} {obj['object_id']}: {exc}")


def _check_single_instance_per_class(pair: AnnotatedPair, errors: list[str]) -> None:
    for side_name, objects in (
        ("before", pair.manual_before_objects),
        ("after", pair.manual_after_objects),
    ):
        class_ids = [obj["class_id"] for obj in objects]
        if len(class_ids) != len(set(class_ids)):
            errors.append(
                f"{pair.pair_id} {side_name}: duplicate class_id in same image "
                f"({sorted(class_ids)})"
            )


def _check_correspondences(pair: AnnotatedPair, errors: list[str]) -> None:
    before_ids = {obj["object_id"] for obj in pair.manual_before_objects}
    after_ids = {obj["object_id"] for obj in pair.manual_after_objects}
    for corr in pair.correspondences:
        if corr["before_object_id"] not in before_ids:
            errors.append(
                f"{pair.pair_id}: correspondence references unknown before "
                f"{corr['before_object_id']!r}"
            )
        if corr["after_object_id"] not in after_ids:
            errors.append(
                f"{pair.pair_id}: correspondence references unknown after "
                f"{corr['after_object_id']!r}"
            )


def _check_group_integrity(
    pairs: dict[str, AnnotatedPair], manifest: SplitManifest, errors: list[str]
) -> None:
    """Every capture_group_id and every shared image must live in exactly one split."""
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


def audit(
    pairs: dict[str, AnnotatedPair],
    manifest: SplitManifest,
    required_class_ids: frozenset[int] = REQUIRED_CLASS_IDS,
) -> AuditReport:
    """Run all Part A checks. Returns a report; never raises for data problems."""
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
        _check_single_instance_per_class(pair, errors)
        _check_correspondences(pair, errors)

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
    image_counts = {name: 0 for name in manifest.splits}
    for name, ids in manifest.splits.items():
        imgs: set[str] = set()
        for pid in ids:
            pair = pairs[pid]
            imgs.add(pair.before_image_id)
            imgs.add(pair.after_image_id)
        image_counts[name] = len(imgs)
    group_counts = {name: 0 for name in manifest.splits}
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
