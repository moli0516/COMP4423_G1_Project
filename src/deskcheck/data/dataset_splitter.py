"""Owner A: whole-group split near 60/20/20.

Related pairs (same capture_group_id or shared image) must stay in one split.
Coverage beats exact ratios: fail clearly if coverage is infeasible.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest

VALID_STATUSES = ("added", "removed", "moved", "unchanged")
TARGET_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}


@dataclass(frozen=True)
class GroupBucket:
    group_id: str
    pair_ids: tuple[str, ...]
    class_ids: frozenset[int]
    statuses: frozenset[str]
    no_change_count: int


def _build_groups(pairs: Iterable[AnnotatedPair]) -> list[GroupBucket]:
    """Collapse pairs into capture-group buckets. Pairs sharing an image are merged."""
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
        buckets.append(
            GroupBucket(
                group_id=root,
                pair_ids=tuple(p.pair_id for p in group_pairs),
                class_ids=frozenset(class_ids),
                statuses=frozenset(statuses),
                no_change_count=no_change,
            )
        )
    return buckets


def _seed_required_statuses(buckets: list[GroupBucket]) -> dict[str, str]:
    """Pick one bucket per split to guarantee val/test status coverage."""
    assignment: dict[str, str] = {}
    remaining = list(buckets)
    required = set(VALID_STATUSES)
    for split_name in ("val", "test"):
        # Prefer the bucket that covers all required statuses and has no-change pairs.
        candidates = [
            b for b in remaining
            if required.issubset(b.statuses) and b.no_change_count > 0
        ]
        if not candidates:
            # Relax: at least needs no-change pairs and as many statuses as possible.
            candidates = [b for b in remaining if b.no_change_count > 0]
            if not candidates:
                raise ValueError(
                    f"cannot guarantee {split_name} no-change coverage: "
                    "no remaining bucket has independent no-change pairs"
                )
        best = max(candidates, key=lambda b: (len(b.statuses), -len(b.pair_ids)))
        if not required.issubset(best.statuses):
            raise ValueError(
                f"cannot guarantee {split_name} status coverage: best bucket "
                f"{best.group_id!r} only has {sorted(best.statuses)}"
            )
        assignment[best.group_id] = split_name
        remaining.remove(best)
    return assignment


def split(
    pairs: list[AnnotatedPair],
    seed: int = 42,
    target_ratios: dict[str, float] = TARGET_RATIOS,
) -> SplitManifest:
    """Assign whole groups to train/val/test. Deterministic for a given seed."""
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

    rng = random.Random(seed)
    assignment = _seed_required_statuses(buckets)
    remaining = [b for b in buckets if b.group_id not in assignment]
    rng.shuffle(remaining)

    target_counts = {
        name: max(round(target_ratios[name] * total_pairs), 1)
        for name in target_ratios
    }
    current = {name: 0 for name in target_ratios}
    for gid, name in assignment.items():
        bucket = next(b for b in buckets if b.group_id == gid)
        current[name] += len(bucket.pair_ids)

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
