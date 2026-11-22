import math
from collections import defaultdict

import pytest

from deskcheck.data.dataset_catalog import AnnotatedPair
from deskcheck.data.dataset_splitter import split

REQUIRED_CLASSES = set(range(6))
REQUIRED_STATUSES = {"added", "removed", "moved", "unchanged"}


def make_pair(pair_id, group_id, classes, *, changed=False):
    classes = set(classes)

    before_classes = classes - ({0} if changed else set())
    after_classes = classes - ({1} if changed else set())

    def objects(side, class_ids):
        result = []
        for class_id in sorted(class_ids):
            x = 20.0 + class_id * 100.0
            if changed and side == "after" and class_id == 2:
                x += 100.0

            result.append(
                {
                    "object_id": f"{pair_id}_{side}_{class_id}",
                    "class_id": class_id,
                    "bbox_xyxy": [x, 100.0, x + 40.0, 160.0],
                }
            )
        return tuple(result)

    before = objects("before", before_classes)
    after = objects("after", after_classes)

    before_by_class = {obj["class_id"]: obj for obj in before}
    after_by_class = {obj["class_id"]: obj for obj in after}

    correspondences = []
    events = []

    for class_id in sorted(classes):
        before_obj = before_by_class.get(class_id)
        after_obj = after_by_class.get(class_id)

        if before_obj is None:
            status = "added"
        elif after_obj is None:
            status = "removed"
        else:
            status = (
                "moved"
                if changed and class_id == 2
                else "unchanged"
            )
            correspondences.append(
                {
                    "class_id": class_id,
                    "before_object_id": before_obj["object_id"],
                    "after_object_id": after_obj["object_id"],
                }
            )

        events.append(
            {
                "class_id": class_id,
                "status": status,
                "before_object_id": (
                    before_obj["object_id"] if before_obj else None
                ),
                "after_object_id": (
                    after_obj["object_id"] if after_obj else None
                ),
            }
        )

    return AnnotatedPair(
        pair_id=pair_id,
        capture_group_id=group_id,
        before_image_id=f"{pair_id}_before",
        after_image_id=f"{pair_id}_after",
        before_image_path=f"synthetic/{pair_id}_before.png",
        after_image_path=f"synthetic/{pair_id}_after.png",
        width=1000,
        height=1000,
        rules_version="b-review-synthetic-v1",
        independently_captured_no_change=not changed,
        manual_before_objects=before,
        manual_after_objects=after,
        correspondences=tuple(correspondences),
        events=tuple(events),
    )


def build_fixture():
    pairs = []

    # Each of these groups independently covers all classes/statuses
    # and contains an independently captured no-change pair.
    for group_id in ("coverage_a", "coverage_b"):
        pairs.append(
            make_pair(
                f"{group_id}_change",
                group_id,
                REQUIRED_CLASSES,
                changed=True,
            )
        )
        pairs.append(
            make_pair(
                f"{group_id}_control",
                group_id,
                REQUIRED_CLASSES,
            )
        )

    # Classes 0–4 are common in the remaining pool.
    for index in range(20):
        pairs.append(
            make_pair(
                f"common_{index:02d}",
                f"common_group_{index:02d}",
                {0, 1, 2, 3, 4},
            )
        )

    # After val/test coverage seeding, this is the only remaining
    # bucket carrying class 5.
    pairs.append(make_pair("rare", "rare_group", {5}))

    return pairs


def assert_pair_semantics(pair):
    """Independent checks for the synthetic annotation fixture."""
    before = {
        obj["class_id"]: obj
        for obj in pair.manual_before_objects
    }
    after = {
        obj["class_id"]: obj
        for obj in pair.manual_after_objects
    }

    assert len(before) == len(pair.manual_before_objects)
    assert len(after) == len(pair.manual_after_objects)

    union = set(before) | set(after)
    assert {event["class_id"] for event in pair.events} == union
    assert len(pair.events) == len(union)

    expected_correspondences = {
        (
            class_id,
            before[class_id]["object_id"],
            after[class_id]["object_id"],
        )
        for class_id in set(before) & set(after)
    }
    actual_correspondences = {
        (
            corr["class_id"],
            corr["before_object_id"],
            corr["after_object_id"],
        )
        for corr in pair.correspondences
    }
    assert actual_correspondences == expected_correspondences
    assert len(pair.correspondences) == len(expected_correspondences)

    tau = 0.02152951369355096
    diagonal = math.hypot(pair.width, pair.height)

    for event in pair.events:
        class_id = event["class_id"]
        before_obj = before.get(class_id)
        after_obj = after.get(class_id)

        assert event["before_object_id"] == (
            before_obj["object_id"] if before_obj else None
        )
        assert event["after_object_id"] == (
            after_obj["object_id"] if after_obj else None
        )

        if before_obj is None:
            expected = "added"
        elif after_obj is None:
            expected = "removed"
        else:
            b = before_obj["bbox_xyxy"]
            a = after_obj["bbox_xyxy"]
            dx = (a[0] + a[2] - b[0] - b[2]) / 2
            dy = (a[1] + a[3] - b[1] - b[3]) / 2
            distance = math.hypot(dx, dy) / diagonal
            expected = "moved" if distance > tau else "unchanged"

        assert event["status"] == expected

    if pair.independently_captured_no_change:
        assert all(
            event["status"] == "unchanged"
            for event in pair.events
        )


def verify_assignment(pairs, splits):
    """Check invariants without calling A's coverage/grouping helpers."""
    assert set(splits) == {"train", "val", "test"}

    pairs_by_id = {pair.pair_id: pair for pair in pairs}
    assert len(pairs_by_id) == len(pairs)

    assigned = [
        pair_id
        for pair_ids in splits.values()
        for pair_id in pair_ids
    ]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(pairs_by_id)

    group_splits = defaultdict(set)
    image_splits = defaultdict(set)
    path_splits = defaultdict(set)

    for split_name, pair_ids in splits.items():
        assert pair_ids, f"{split_name}: empty"

        classes = set()
        statuses = set()
        no_change_count = 0

        for pair_id in pair_ids:
            pair = pairs_by_id[pair_id]
            group_splits[pair.capture_group_id].add(split_name)

            for image_id, image_path in (
                (pair.before_image_id, pair.before_image_path),
                (pair.after_image_id, pair.after_image_path),
            ):
                image_splits[image_id].add(split_name)
                path_splits[image_path].add(split_name)

            for obj in (
                *pair.manual_before_objects,
                *pair.manual_after_objects,
            ):
                classes.add(obj["class_id"])

            statuses.update(event["status"] for event in pair.events)
            no_change_count += int(pair.independently_captured_no_change)

        assert REQUIRED_CLASSES <= classes, (
            f"{split_name}: missing classes {REQUIRED_CLASSES - classes}"
        )

        if split_name in {"val", "test"}:
            assert REQUIRED_STATUSES <= statuses
            assert no_change_count >= 1

    assert all(len(names) == 1 for names in group_splits.values())
    assert all(len(names) == 1 for names in image_splits.values())
    assert all(len(names) == 1 for names in path_splits.values())


def witness_assignment(pairs):
    # 25 pairs total: 15 train / 6 val / 4 test.
    # Whole groups, full coverage, close to the default 60/20/20.
    train_ids = {
        "rare",
        *(f"common_{index:02d}" for index in range(14)),
    }

    splits = {"train": [], "val": [], "test": []}

    for pair in pairs:
        if pair.pair_id in train_ids:
            destination = "train"
        elif pair.capture_group_id == "coverage_a":
            destination = "val"
        elif pair.capture_group_id == "coverage_b":
            destination = "test"
        else:
            index = int(pair.pair_id.rsplit("_", 1)[1])
            destination = "val" if index < 18 else "test"

        splits[destination].append(pair.pair_id)

    return {
        name: tuple(sorted(pair_ids))
        for name, pair_ids in splits.items()
    }


def test_fixture_has_valid_semantics_and_feasible_witness():
    pairs = build_fixture()

    for pair in pairs:
        assert_pair_semantics(pair)

    witness = witness_assignment(pairs)
    verify_assignment(pairs, witness)

    assert {
        name: len(pair_ids)
        for name, pair_ids in witness.items()
    } == {"train": 15, "val": 6, "test": 4}


@pytest.mark.parametrize("seed", range(32))
def test_public_split_default_ratios_preserves_train_coverage(seed):
    pairs = build_fixture()

    # Prove feasibility independently before exercising the splitter.
    verify_assignment(pairs, witness_assignment(pairs))

    try:
        # No patching and no custom target_ratios.
        manifest = split(pairs, seed=seed)
    except ValueError as exc:
        pytest.fail(
            f"seed={seed}: public split() rejected a fixture with a "
            f"verified feasible witness: {exc}"
        )

    verify_assignment(pairs, manifest.splits)

    repeated = split(pairs, seed=seed)
    assert repeated.splits == manifest.splits