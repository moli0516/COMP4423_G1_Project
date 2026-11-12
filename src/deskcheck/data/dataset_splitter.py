"""Owner A: whole-group split near 60/20/20.

Coverage rules:
- val and test each receive all 4 statuses, all required classes, and ≥1 no-change pair.
- train must also receive all required classes.
- Related pairs (same capture_group_id or shared image) stay in the same split.

Coverage may come from multiple buckets per split. Fails loudly if any
coverage requirement is infeasible.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest

VALID_STATUSES = ("added", "removed", "moved", "unchanged")
TARGET_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}
REQUIRED_CLASS_IDS = frozenset(range(6))


@dataclass(frozen=True)
class GroupBucket:
    group_id: str
    pair_ids: tuple[str, ...]
    class_ids: frozenset[int]
    statuses: frozenset[str]
    no_change_count: int


def _build_groups(pairs: Iterable[AnnotatedPair]) -> list[GroupBucket]:
    pairs = list(pairs)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in pairs:
        parent.setdefault(pair.capture_group_id, pair.capture_group_id)
        parent.setdefault(pair.before_image_id, pair.before_image_id)
        parent.setdefault(pair.after_image_id, pair.after_image_id)
        union(pair.capture_group_id, pair.before_image_id)
        union(pair.capture_group_id, pair.after_image_id)

    grouped: dict[str, list[AnnotatedPair]] = {}
    for pair in pairs:
        root = find(pair.capture_group_id)
        grouped.setdefault(root, []).append(pair)

    buckets: list[GroupBucket] = []
    for root, group_pairs in grouped.items():
        class_ids: set[int] = set()
        statuses: set[str] = set()
        no_change = 0
        for pair in group_pairs:
            for obj in (*pair.manual_before_objects, *pair.manual_after_objects):
                class_ids.add(obj["class_id"])
            for ev in pair.events:
                statuses.add(ev["status"])
            if pair.independently_captured_no_change:
                no_change += 1
        buckets.append(GroupBucket(
            group_id=root,
            pair_ids=tuple(p.pair_id for p in group_pairs),
            class_ids=frozenset(class_ids),
            statuses=frozenset(statuses),
            no_change_count=no_change,
        ))
    return buckets


def _coverage_failure_reason(
    buckets: list[GroupBucket], required_statuses: set[str],
    required_classes: set[int],
) -> str:
    """Describe why coverage is infeasible, distinguishing proven from greedy-gap.

    Proven infeasible: even taking EVERY remaining bucket, coverage is short.
    Greedy gap:       union of remaining buckets has coverage, but no single
                      or small set was found by the greedy picks.
    """
    union_statuses: set[str] = set()
    union_classes: set[int] = set()
    has_no_change = False
    for b in buckets:
        union_statuses |= b.statuses
        union_classes |= b.class_ids
        if b.no_change_count > 0:
            has_no_change = True

    missing_statuses = required_statuses - union_statuses
    missing_classes = required_classes - union_classes
    if missing_statuses or missing_classes or not has_no_change:
        return (
            f"infeasible: union of all groups still misses "
            f"statuses={sorted(missing_statuses)}, "
            f"classes={sorted(missing_classes)}, "
            f"no_change_pair={'yes' if has_no_change else 'no'}"
        )
    return (
        "greedy search could not assemble a covering set even though the "
        "union of remaining buckets has full coverage"
    )


def _assign_coverage_targets(
    buckets: list[GroupBucket],
    targets: tuple[str, ...] = ("val", "test"),
) -> dict[str, str]:
    """Assign whole buckets to coverage targets (val, test).

    Each target receives all 4 statuses, all required classes, and ≥1
    no-change pair. Coverage may come from multiple buckets. Buckets
    selected for one target are removed from the pool for the next target.
    """
    required_statuses = set(VALID_STATUSES)
    required_classes = set(REQUIRED_CLASS_IDS)
    remaining = list(buckets)
    assignment: dict[str, str] = {}

    for target in targets:
        selected: list[GroupBucket] = []
        union_statuses: set[str] = set()
        union_classes: set[int] = set()

        # First: pick a bucket with a no-change pair (rarest property).
        no_change_candidates = [b for b in remaining if b.no_change_count > 0]
        if not no_change_candidates:
            reason = _coverage_failure_reason(
                remaining, required_statuses, required_classes
            )
            raise ValueError(
                f"cannot guarantee {target} coverage: no remaining bucket "
                f"has an independent no-change pair ({reason})"
            )
        no_change_pick = max(
            no_change_candidates,
            key=lambda b: (len(b.statuses), len(b.class_ids)),
        )
        selected.append(no_change_pick)
        union_statuses |= no_change_pick.statuses
        union_classes |= no_change_pick.class_ids
        remaining.remove(no_change_pick)

        # Greedy fill: add buckets until statuses + classes are covered.
        while (not (required_statuses <= union_statuses
                    and required_classes <= union_classes)) and remaining:
            def _score(b: GroupBucket,
                       cur_statuses: set[str] = union_statuses,
                       cur_classes: set[int] = union_classes) -> int:
                new_s = len((b.statuses - cur_statuses) & required_statuses)
                new_c = len((b.class_ids - cur_classes) & required_classes)
                return new_s + new_c

            pick = max(remaining, key=_score)
            if _score(pick) == 0:
                break
            selected.append(pick)
            union_statuses |= pick.statuses
            union_classes |= pick.class_ids
            remaining.remove(pick)

        missing_statuses = required_statuses - union_statuses
        missing_classes = required_classes - union_classes
        if missing_statuses or missing_classes:
            reason = _coverage_failure_reason(
                remaining + selected, required_statuses, required_classes
            )
            raise ValueError(
                f"cannot guarantee {target} coverage: "
                f"missing statuses={sorted(missing_statuses)}, "
                f"missing classes={sorted(missing_classes)} ({reason})"
            )

        for b in selected:
            assignment[b.group_id] = target

    return assignment


def _ensure_train_class_coverage(
    buckets: list[GroupBucket],
    assignment: dict[str, str],
    required_classes: set[int],
) -> None:
    """Verify train (buckets not assigned to val/test) covers all required classes.

    If train is missing classes, attempt to move small val/test buckets back
    to train without breaking val/test coverage. If impossible, raise.
    """
    train_bucket_ids = [b.group_id for b in buckets if b.group_id not in assignment]
    train_classes: set[int] = set()
    for b in buckets:
        if b.group_id in train_bucket_ids:
            train_classes |= b.class_ids

    missing = required_classes - train_classes
    if not missing:
        return

    # Try moving small buckets out of val/test back into train to fill gaps.
    # Only move buckets whose removal doesn't strip val/test of their required
    # coverage (checked by simulating the removal).
    moveable = sorted(
        [b for b in buckets if b.group_id in assignment],
        key=lambda b: (len(b.pair_ids), -len(b.class_ids & missing)),
    )

    for candidate in moveable:
        if not (candidate.class_ids & missing):
            continue
        # Would moving this bucket break its current split's coverage?
        remaining_in_split = [
            b for b in buckets
            if assignment.get(b.group_id) == assignment[candidate.group_id]
            and b.group_id != candidate.group_id
        ]
        split_statuses: set[str] = set()
        split_classes: set[int] = set()
        split_has_no_change = False
        for b in remaining_in_split:
            split_statuses |= b.statuses
            split_classes |= b.class_ids
            if b.no_change_count > 0:
                split_has_no_change = True
        if not (required_classes <= split_classes):
            continue  # would break class coverage in val/test
        if not (set(VALID_STATUSES) <= split_statuses):
            continue  # would break status coverage in val/test
        if not split_has_no_change:
            continue  # would break no-change coverage in val/test

        # Safe to move
        train_bucket_ids.append(candidate.group_id)
        train_classes |= candidate.class_ids
        del assignment[candidate.group_id]
        missing = required_classes - train_classes
        if not missing:
            return

    raise ValueError(
        f"cannot guarantee train class coverage: missing classes "
        f"{sorted(missing)} even after reassigning from val/test. "
        f"Collect more data so every class appears in more groups."
    )


def split(
    pairs: list[AnnotatedPair],
    seed: int = 42,
    target_ratios: dict[str, float] | None = None,
) -> SplitManifest:
    """Assign whole groups to train/val/test. Deterministic for a given seed."""
    if target_ratios is None:
        target_ratios = dict(TARGET_RATIOS)
    if not pairs:
        raise ValueError("cannot split empty dataset")
    if abs(sum(target_ratios.values()) - 1.0) > 1e-6:
        raise ValueError("target_ratios must sum to 1.0")
    if set(target_ratios.keys()) != {"train", "val", "test"}:
        raise ValueError("target_ratios must have exactly train, val, test keys")

    buckets = _build_groups(pairs)
    total_pairs = sum(len(b.pair_ids) for b in buckets)
    if total_pairs == 0:
        raise ValueError("no pairs to split")

    assignment = _assign_coverage_targets(buckets)

    # NEW: guarantee train has all required classes, moving buckets if needed.
    _ensure_train_class_coverage(
        buckets, assignment, set(REQUIRED_CLASS_IDS)
    )

    remaining = [b for b in buckets if b.group_id not in assignment]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    target_counts = {
        name: max(round(target_ratios[name] * total_pairs), 1)
        for name in target_ratios
    }
    current = {name: 0 for name in target_ratios}
    for bucket in buckets:
        target = assignment.get(bucket.group_id)
        if target is not None:
            current[target] += len(bucket.pair_ids)

    for bucket in remaining:
        best_split = min(
            target_ratios,
            key=lambda name: (
                (current[name] + len(bucket.pair_ids)) / target_counts[name]
            ),
        )
        assignment[bucket.group_id] = best_split
        current[best_split] += len(bucket.pair_ids)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for bucket in buckets:
        splits[assignment[bucket.group_id]].extend(bucket.pair_ids)

    for name, ids in splits.items():
        if not ids:
            raise ValueError(f"split {name!r} ended up empty after grouping")

    return SplitManifest(
        schema_version="1.0",
        seed=seed,
        dataset_release="v1",
        splits={name: tuple(sorted(ids)) for name, ids in splits.items()},
        coverage={},
    )
