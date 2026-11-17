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


# TODO(part-a): add a direct unit test for _coverage_failure_reason once
# the splitter's coverage-seeding logic stabilizes. The 'infeasible' branch
# is exercised indirectly during development; the 'greedy gap' branch is
# currently untested. If B asks, a small fixture can test this directly.

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
            f"union of remaining buckets misses "
            f"statuses={sorted(missing_statuses)}, "
            f"classes={sorted(missing_classes)}, "
            f"no_change_pair={'yes' if has_no_change else 'no'}; "
            f"global infeasibility not proven"
        )
    return (
        "greedy search could not assemble a covering set even though the "
        "union of remaining buckets has full coverage; a covering subset "
        "may exist"
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

    # Coverage-preserving allocation:
    # 1. Compute which required classes are not yet represented in the pool
    #    of buckets that will be assigned to train.
    # 2. Force any bucket that fills a train gap into train first.
    # 3. Then assign the remaining buckets by ratio.
    # 4. Verify train ends with all required classes.

    unassigned = [b for b in buckets if b.group_id not in assignment]
    train_candidate_classes: set[int] = set()
    for b in unassigned:
        train_candidate_classes |= b.class_ids
    missing_train = set(REQUIRED_CLASS_IDS) - train_candidate_classes

    target_counts = {
        name: max(round(target_ratios[name] * total_pairs), 1)
        for name in target_ratios
    }
    current = {name: 0 for name in target_ratios}
    for bucket in buckets:
        target = assignment.get(bucket.group_id)
        if target is not None:
            current[target] += len(bucket.pair_ids)

    # Sort: gap-filling buckets first, then by pair count (smallest first, so
    # they're easy to slot without breaking ratio balance).
    rng = random.Random(seed)
    rng.shuffle(unassigned)
    unassigned.sort(
        key=lambda b: (0 if (b.class_ids & missing_train) else 1, len(b.pair_ids))
    )

    for bucket in unassigned:
        if bucket.class_ids & missing_train:
            # Force into train to preserve coverage.
            assignment[bucket.group_id] = "train"
            current["train"] += len(bucket.pair_ids)
            missing_train -= bucket.class_ids
            continue
        # Otherwise assign by ratio.
        best_split = min(
            target_ratios,
            key=lambda name: (
                (current[name] + len(bucket.pair_ids)) / target_counts[name]
            ),
        )
        assignment[bucket.group_id] = best_split
        current[best_split] += len(bucket.pair_ids)

    # Final verification: train actually has all required classes.
    final_train_classes: set[int] = set()
    for b in buckets:
        if assignment.get(b.group_id) == "train":
            final_train_classes |= b.class_ids
    missing_after = set(REQUIRED_CLASS_IDS) - final_train_classes
    if missing_after:
        raise ValueError(
            f"cannot guarantee train class coverage after allocation: "
            f"missing {sorted(missing_after)}. Collect more data so every "
            f"class appears in more groups."
        )

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
