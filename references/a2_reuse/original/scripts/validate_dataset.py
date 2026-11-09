#!/usr/bin/env python3
"""Read-only dataset validator for COMP4423 Assignment 2.

Save as scripts/validate_dataset.py. It never modifies images or CSV files.
Requires Pillow: python -m pip install Pillow
"""

import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
import sys

from PIL import Image, UnidentifiedImageError

IMAGE_HEADERS = ["image_id", "relative_path", "label", "group_id", "session_id", "scene_type", "lighting", "split"]
LOCATION_HEADERS = ["group_id", "class_name", "location_description", "boundary_verified"]
CLASS_LABELS = {"inside": "1", "outside": "0"}
VALID_SPLITS = {"train", "validation", "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VERIFIED_VALUES = {"yes", "pending"}


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)


def read_csv(path, required_headers, report):
    if not path.is_file():
        report.error(f"Missing file: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                report.error(f"{path.name}: file is empty or has no header.")
                return []
            headers = [name.strip() for name in reader.fieldnames]
            if headers != required_headers:
                report.error(f"{path.name}: expected header {required_headers}, found {headers}.")
                return []
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    report.error(f"{path.name}:{line_number}: too many columns.")
                    continue
                cleaned = {key.strip(): (value or "").strip() for key, value in row.items()}
                cleaned["_line"] = line_number
                rows.append(cleaned)
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        report.error(f"Could not read {path}: {error}")
        return []


def safe_relative_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    return bool(value) and not path.is_absolute() and ".." not in path.parts and len(path.parts) == 4 and path.parts[0] == "images"


def validate_locations(rows, report):
    locations = {}
    for row in rows:
        group_id, class_name = row["group_id"], row["class_name"]
        place = f"locations.csv:{row['_line']}"
        if not group_id:
            report.error(f"{place}: empty group_id.")
            continue
        if group_id in locations:
            report.error(f"{place}: duplicate group_id {group_id}.")
            continue
        expected_prefix = "in_" if class_name == "inside" else "out_"
        if class_name not in CLASS_LABELS:
            report.error(f"{place}: class_name must be inside or outside.")
        elif not group_id.startswith(expected_prefix) or not group_id[len(expected_prefix):].isdigit():
            report.error(f"{place}: {group_id} does not match {class_name} naming.")
        if not row["location_description"]:
            report.error(f"{place}: location_description is empty for {group_id}.")
        if row["boundary_verified"].lower() not in VERIFIED_VALUES:
            report.error(f"{place}: boundary_verified must be yes or pending.")
        locations[group_id] = row
    return locations


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_images(dataset_dir, rows, locations, report, min_side):
    seen_ids, seen_paths = {}, {}
    hashes = {}
    valid_rows = []
    for row in rows:
        place = f"images.csv:{row['_line']}"
        image_id, relative_path = row["image_id"], row["relative_path"].replace("\\", "/")
        if not image_id:
            report.error(f"{place}: empty image_id.")
        elif image_id in seen_ids:
            report.error(f"{place}: duplicate image_id {image_id}; first seen on line {seen_ids[image_id]}.")
        else:
            seen_ids[image_id] = row["_line"]
        if relative_path in seen_paths:
            report.error(f"{place}: duplicate relative_path {relative_path}.")
        else:
            seen_paths[relative_path] = row["_line"]
        if not safe_relative_path(relative_path):
            report.error(f"{place}: path must look like images/<inside|outside>/<group_id>/<file>.")
            continue

        parts = PurePosixPath(relative_path).parts
        class_name, folder_group = parts[1], parts[2]
        if class_name not in CLASS_LABELS:
            report.error(f"{place}: folder class must be inside or outside.")
            continue
        if row["label"] not in {"0", "1"}:
            report.error(f"{place}: label must be 0 or 1.")
        elif row["label"] != CLASS_LABELS[class_name]:
            report.error(f"{place}: label {row['label']} conflicts with {class_name} folder.")
        if row["group_id"] != folder_group:
            report.error(f"{place}: group_id {row['group_id']} conflicts with folder {folder_group}.")
        location = locations.get(row["group_id"])
        if location is None:
            report.error(f"{place}: group {row['group_id']} is missing from locations.csv.")
        elif location["class_name"] != class_name:
            report.error(f"{place}: locations.csv class conflicts with folder for {row['group_id']}.")
        if row["split"] and row["split"] not in VALID_SPLITS:
            report.error(f"{place}: split must be blank, train, validation, or test.")

        path = dataset_dir / Path(*parts)
        if not path.is_file():
            report.error(f"{place}: file does not exist: {relative_path}")
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            report.error(f"{place}: unsupported image extension {path.suffix}.")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            report.error(f"{place}: cannot decode image {relative_path}: {error}")
            continue
        if min(width, height) < min_side:
            report.warn(f"{relative_path}: small image {width}x{height}; check whether enough scene detail remains.")
        digest = file_hash(path)
        hashes.setdefault(digest, []).append(row)
        valid_rows.append(row)

    for duplicates in hashes.values():
        if len(duplicates) < 2:
            continue
        paths = ", ".join(item["relative_path"] for item in duplicates)
        groups = {item["group_id"] for item in duplicates}
        splits = {item["split"] for item in duplicates if item["split"]}
        if len(groups) > 1 or len(splits) > 1:
            report.error(f"Identical file content appears across groups/splits: {paths}")
        else:
            report.warn(f"Exact duplicate files in one group: {paths}")
    return valid_rows


def find_unlisted_files(dataset_dir, rows, report):
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        report.error(f"Missing image directory: {images_dir}")
        return
    listed = {row["relative_path"].replace("\\", "/") for row in rows}
    for path in sorted(images_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(images_dir).parts):
            continue
        relative = path.relative_to(dataset_dir).as_posix()
        if path.suffix.lower() in IMAGE_EXTENSIONS and relative not in listed:
            report.error(f"Image exists but is not listed in images.csv: {relative}")
        elif path.suffix.lower() in {".heic", ".heif"}:
            report.warn(f"Unsupported camera format ignored; convert before use: {relative}")


def validate_splits(rows, require_split, report):
    group_splits = {}
    for row in rows:
        if row["split"]:
            group_splits.setdefault(row["group_id"], set()).add(row["split"])
    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            report.error(f"Group {group_id} is split across {sorted(splits)}.")
    missing = sorted({row["group_id"] for row in rows if not row["split"]})
    if require_split and missing:
        report.error("Images without split in groups: " + ", ".join(missing))
    elif missing:
        report.warn("Split not assigned yet for groups: " + ", ".join(missing))
    if require_split or (rows and not missing):
        for split in sorted(VALID_SPLITS):
            labels = {row["label"] for row in rows if row["split"] == split}
            if labels != {"0", "1"}:
                report.error(f"Split {split} must contain both classes; found labels {sorted(labels)}.")


def summarize(rows, locations, report):
    verified_groups = {gid for gid, row in locations.items() if row["boundary_verified"].lower() == "yes"}
    used_groups = {row["group_id"] for row in rows}
    for group_id in sorted(used_groups - verified_groups):
        report.warn(f"Group {group_id} is not boundary-verified; do not treat its labels as final.")
    for group_id in sorted(set(locations) - used_groups):
        report.warn(f"Location {group_id} has no image rows yet.")
    print("\nDataset summary")
    print("class     images  groups")
    for class_name, label in CLASS_LABELS.items():
        class_rows = [row for row in rows if row["label"] == label]
        print(f"{class_name:<9} {len(class_rows):>6}  {len({row['group_id'] for row in class_rows}):>6}")
    if any(row["split"] for row in rows):
        print("\nSplit summary: images / groups")
        for split in ["train", "validation", "test"]:
            parts = []
            for class_name, label in CLASS_LABELS.items():
                selected = [row for row in rows if row["split"] == split and row["label"] == label]
                parts.append(f"{class_name}={len(selected)}/{len({row['group_id'] for row in selected})}")
            print(f"{split:<10} " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent.parent / "dataset")
    parser.add_argument("--require-verified", action="store_true", help="Treat unverified groups as errors.")
    parser.add_argument("--require-split", action="store_true", help="Require complete group-level splits.")
    parser.add_argument("--min-side", type=int, default=128, help="Warn if the shortest image side is smaller.")
    args = parser.parse_args()

    report = Report()
    dataset_dir = args.dataset_dir.resolve()
    location_rows = read_csv(dataset_dir / "locations.csv", LOCATION_HEADERS, report)
    image_rows = read_csv(dataset_dir / "images.csv", IMAGE_HEADERS, report)
    locations = validate_locations(location_rows, report)
    valid_rows = validate_images(dataset_dir, image_rows, locations, report, args.min_side)
    find_unlisted_files(dataset_dir, image_rows, report)
    validate_splits(valid_rows, args.require_split, report)
    summarize(valid_rows, locations, report)

    if args.require_verified:
        report.errors.extend(message for message in report.warnings if "not boundary-verified" in message)
        report.warnings = [message for message in report.warnings if "not boundary-verified" not in message]

    print(f"\nChecked {len(image_rows)} image rows and {len(location_rows)} location rows in {dataset_dir}")
    for message in report.warnings:
        print(f"WARNING: {message}")
    for message in report.errors:
        print(f"ERROR: {message}", file=sys.stderr)
    print("\nOnly file/metadata consistency was checked. This does not verify real GPS location, privacy, or hidden same-scene overlap.")
    if report.errors:
        print(f"Validation FAILED: {len(report.errors)} error(s), {len(report.warnings)} warning(s).", file=sys.stderr)
        return 1
    print(f"Validation passed: 0 errors, {len(report.warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())