from unittest.mock import patch

import pytest

from deskcheck.data import dataset_splitter as ds


def bucket(group_id, pair_count, classes):
    return ds.GroupBucket(
        group_id=group_id,
        pair_ids=tuple(f"{group_id}_p{i}" for i in range(pair_count)),
        class_ids=frozenset(classes),
        statuses=frozenset(ds.VALID_STATUSES),
        no_change_count=1,
    )


def test_train_coverage_reserved_before_ratio_assignment():
    required = set(range(6))

    buckets = [
        bucket("val_seed", 1, required),
        bucket("test_seed", 1, required),
        bucket("common", 1, {0, 1, 2, 3, 4}),
        bucket("rare", 1, {5}),
        bucket("filler", 6, {0, 1, 2, 3, 4}),
    ]

    seeded_assignment = {
        "val_seed": "val",
        "test_seed": "test",
    }

    classes_by_pair = {
        pair_id: set(group.class_ids)
        for group in buckets
        for pair_id in group.pair_ids
    }

    # A feasible assignment exists:
    # train = common + rare; val/test retain their full-coverage seeds.
    ratios = {"train": 0.2, "val": 0.4, "test": 0.4}

    for seed in range(10):
        with (
            patch.object(ds, "_build_groups", return_value=buckets),
            patch.object(
                ds,
                "_assign_coverage_targets",
                return_value=dict(seeded_assignment),
            ),
        ):
            try:
                # Pair contents are bypassed by the patched grouping seam.
                manifest = ds.split(
                    [object()],
                    seed=seed,
                    target_ratios=ratios,
                )
            except ValueError as exc:
                pytest.fail(
                    f"seed={seed}: feasible train coverage was not reserved: "
                    f"{exc}"
                )

        for split_name, pair_ids in manifest.splits.items():
            observed = set()
            for pair_id in pair_ids:
                observed |= classes_by_pair[pair_id]

            assert required <= observed, (
                f"seed={seed}, split={split_name}, "
                f"missing={required - observed}"
            )