#!/usr/bin/env python3
"""Assign train/validation/test splits by location group for COMP4423 Assignment 2.

Save as scripts/split_dataset.py. Whole location groups are assigned to one split,
separately within inside and outside, targeting 60% / 20% / 20% of groups.

The optional seeded search compares random group permutations using image counts only.
It never looks at features, predictions, accuracy, or test results.
Standard library only.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import tempfile

SPLITS = ("train", "validation", "test")
TARGET_SHARE = {"train": 0.6, "validation": 0.2, "test": 0.2}
CLASS_BY_LABEL = {"1": "inside", "0": "outside"}
REQUIRED_IMAGE_COLUMNS = {"image_id", "relative_path", "label", "group_id", "split"}
REQUIRED_LOCATION_COLUMNS = {"group_id", "class_name", "boundary_verified"}
MIN_GROUPS_PER_CLASS = 3


class SplitError(ValueError):
    """Raised when the dataset cannot be split safely."""


def read_csv_table(path, required_columns):
    if not path.is_file():
        raise SplitError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SplitError(f"{path.name} is empty or has no header.")
        fieldnames = [name.strip() for name in reader.fieldnames]
        missing = required_columns - set(fieldnames)
        if missing:
            raise SplitError(f"{path.name} is missing columns: {sorted(missing)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise SplitError(f"{path.name}:{line_number}: too many columns.")
            cleaned = {key.strip(): (value or "").strip() for key, value in row.items()}
            cleaned["_line"] = line_number
            rows.append(cleaned)
    return fieldnames, rows


def read_locations(path):
    _, rows = read_csv_table(path, REQUIRED_LOCATION_COLUMNS)
    locations = {}
    for row in rows:
        group_id = row["group_id"]
        if not group_id:
            raise SplitError(f"locations.csv:{row['_line']}: empty group_id.")
        if group_id in locations:
            raise SplitError(f"locations.csv:{row['_line']}: duplicate group_id {group_id}.")
        locations[group_id] = row
    return locations


def group_counts(group_total):
    """Return (train, validation, test) group counts targeting 60/20/20."""
    if group_total < MIN_GROUPS_PER_CLASS:
        raise SplitError(f"At least {MIN_GROUPS_PER_CLASS} groups are needed; found {group_total}.")
    validation = max(1, (group_total * 20 + 50) // 100)
    test = max(1, (group_total * 20 + 50) // 100)
    train = group_total - validation - test
    if train < 1:
        raise SplitError("Not enough groups to keep train, validation, and test non-empty.")
    return train, validation, test


def collect_groups(image_rows, locations, allow_unverified, force):
    """Return {class_name: {group_id: image_count}} and warnings after strict checks."""
    if not image_rows:
        raise SplitError("images.csv has no image rows.")
    group_labels, group_images, warnings = {}, {}, []
    for row in image_rows:
        place = f"images.csv:{row['_line']}"
        for column in ("image_id", "relative_path", "label", "group_id"):
            if not row[column]:
                raise SplitError(f"{place}: empty {column}.")
        if row["label"] not in CLASS_BY_LABEL:
            raise SplitError(f"{place}: label must be 0 or 1.")
        if row["split"] and not force:
            raise SplitError(
                "images.csv already contains split assignments. Keep the frozen split, or use --force "
                "only if you deliberately want to reassign every group."
            )
        if row["split"] and row["split"] not in SPLITS:
            raise SplitError(f"{place}: existing split must be blank, train, validation, or test.")
        group_id = row["group_id"]
        if group_labels.setdefault(group_id, row["label"]) != row["label"]:
            raise SplitError(f"{place}: group {group_id} contains both inside and outside labels.")
        group_images[group_id] = group_images.get(group_id, 0) + 1

    groups = {"inside": {}, "outside": {}}
    for group_id, label in sorted(group_labels.items()):
        class_name = CLASS_BY_LABEL[label]
        location = locations.get(group_id)
        if location is None:
            raise SplitError(f"Group {group_id} is missing from locations.csv.")
        if location["class_name"] != class_name:
            raise SplitError(f"Group {group_id}: locations.csv says {location['class_name']} but labels say {class_name}.")
        verified = location["boundary_verified"].strip().lower() == "yes"
        if not verified and not allow_unverified:
            raise SplitError(
                f"Group {group_id} is not boundary-verified. Verify it, or use --allow-unverified "
                "only for a throwaway test split."
            )
        if not verified:
            warnings.append(f"Group {group_id} is not boundary-verified; this split must not be treated as final.")
        groups[class_name][group_id] = group_images[group_id]

    for class_name, class_groups in groups.items():
        if len(class_groups) < MIN_GROUPS_PER_CLASS:
            raise SplitError(
                f"{class_name} has {len(class_groups)} group(s); at least {MIN_GROUPS_PER_CLASS} are needed "
                "so train, validation, and test can each contain both classes."
            )
    return groups, warnings


def assign_class(group_ids, counts, rng):
    ordered = sorted(group_ids)
    rng.shuffle(ordered)
    train, validation, _ = counts
    return {
        "train": ordered[:train],
        "validation": ordered[train:train + validation],
        "test": ordered[train + validation:],
    }


def balance_score(assignment, images_by_group):
    total = sum(images_by_group[group_id] for groups in assignment.values() for group_id in groups)
    score = 0.0
    for split in SPLITS:
        share = sum(images_by_group[group_id] for group_id in assignment[split]) / total
        score += abs(share - TARGET_SHARE[split])
    return score


def choose_assignment(groups_by_class, seed, trials):
    """Return {group_id: split}. The search uses group image counts only."""
    if trials < 1:
        raise SplitError("--trials must be at least 1.")
    rng = random.Random(seed)
    counts = {name: group_counts(len(groups)) for name, groups in groups_by_class.items()}
    best, best_score = None, None
    for _ in range(trials):
        candidate, score = {}, 0.0
        for class_name in sorted(groups_by_class):
            class_assignment = assign_class(groups_by_class[class_name], counts[class_name], rng)
            score += balance_score(class_assignment, groups_by_class[class_name])
            for split, group_ids in class_assignment.items():
                candidate.update({group_id: split for group_id in group_ids})
        if best_score is None or score < best_score - 1e-12:
            best, best_score = candidate, score
    return best, best_score, counts


def build_summary(rows, assignment):
    summary = {}
    for class_name, label in (("inside", "1"), ("outside", "0")):
        summary[class_name] = {}
        class_rows = [row for row in rows if row["label"] == label]
        for split in SPLITS:
            selected = [row for row in class_rows if assignment[row["group_id"]] == split]
            summary[class_name][split] = {
                "images": len(selected),
                "groups": len({row["group_id"] for row in selected}),
            }
    return summary


def print_summary(summary):
    print("\nSplit summary (images / groups)")
    print(f"{'class':<9}" + "".join(f"{split:>16}" for split in SPLITS) + f"{'total':>16}")
    for class_name, values in summary.items():
        total_images = sum(item["images"] for item in values.values())
        total_groups = sum(item["groups"] for item in values.values())
        cells = []
        for split in SPLITS:
            item = values[split]
            share = 100 * item["images"] / total_images if total_images else 0
            cells.append(f"{item['images']}/{item['groups']} ({share:.0f}%)")
        print(f"{class_name:<9}" + "".join(f"{cell:>16}" for cell in cells) + f"{f'{total_images}/{total_groups}':>16}")


def write_text_atomic(path, writer):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def split_dataset(dataset_dir, seed=42, trials=500, allow_unverified=False, force=False, dry_run=False):
    dataset_dir = Path(dataset_dir).resolve()
    fieldnames, image_rows = read_csv_table(dataset_dir / "images.csv", REQUIRED_IMAGE_COLUMNS)
    locations = read_locations(dataset_dir / "locations.csv")
    groups, warnings = collect_groups(image_rows, locations, allow_unverified, force)
    assignment, score, counts = choose_assignment(groups, seed, trials)
    for row in image_rows:
        row["split"] = assignment[row["group_id"]]
    summary = build_summary(image_rows, assignment)

    report = {
        "seed": seed,
        "trials": trials,
        "target_group_share": TARGET_SHARE,
        "group_counts_per_class": {name: dict(zip(SPLITS, values)) for name, values in counts.items()},
        "image_balance_score": round(score, 6),
        "summary": summary,
        "groups": {split: sorted(group_id for group_id, value in assignment.items() if value == split) for split in SPLITS},
        "warnings": warnings,
        "note": "images.csv is the authoritative split. This report documents how it was generated.",
    }
    if not dry_run:
        def write_images(handle):
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(image_rows)

        write_text_atomic(dataset_dir / "images.csv", write_images)
        write_text_atomic(dataset_dir / "split_report.json", lambda handle: json.dump(report, handle, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent.parent / "dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible assignment.")
    parser.add_argument("--trials", type=int, default=500, help="Random group permutations compared by image balance only; 1 disables search.")
    parser.add_argument("--allow-unverified", action="store_true", help="Allow groups whose boundary_verified is not yes.")
    parser.add_argument("--force", action="store_true", help="Reassign even when splits already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Show the proposed split without writing files.")
    args = parser.parse_args()
    try:
        report = split_dataset(args.dataset_dir, args.seed, args.trials, args.allow_unverified, args.force, args.dry_run)
    except (OSError, SplitError, csv.Error, UnicodeDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print_summary(report["summary"])
    for split in SPLITS:
        print(f"{split}: {', '.join(report['groups'][split])}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    if args.dry_run:
        print("\nDry run only: no files were changed.")
    else:
        print(f"\nUpdated {args.dataset_dir.resolve() / 'images.csv'} and split_report.json")
        print("Freeze this split before model comparison. Do not redraw it after viewing test results.")
    print("Group-level counts target 60/20/20; image shares can differ because groups have different sizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())