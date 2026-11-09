"""Scan dataset/images and atomically overwrite images.csv.

Save this script as scripts/update_images_csv.py under the project root.
Uses only the Python standard library. Existing metadata is preserved by path.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile

HEADERS = [
    "image_id", "relative_path", "label", "group_id",
    "session_id", "scene_type", "lighting", "split",
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_LABELS = {"inside": "1", "outside": "0"}
METADATA_FIELDS = ("session_id", "scene_type", "lighting", "split")


def read_existing(path):
    if not path.exists() or path.stat().st_size == 0:
        return {}
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise ValueError("Existing images.csv has no relative_path column.")
        for row in reader:
            relative_path = (row.get("relative_path") or "").strip().replace("\\", "/")
            if not relative_path:
                continue
            if relative_path in result:
                raise ValueError(f"Duplicate existing CSV path: {relative_path}")
            result[relative_path] = row
    return result


def read_locations(path):
    if not path.exists() or path.stat().st_size == 0:
        return {}
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"group_id", "class_name", "boundary_verified"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("locations.csv must contain group_id, class_name, boundary_verified.")
        for row in reader:
            group_id = (row.get("group_id") or "").strip()
            if not group_id:
                continue
            if group_id in result:
                raise ValueError(f"Duplicate location group: {group_id}")
            result[group_id] = row
    return result


def scan_dataset(dataset_dir, require_verified=False):
    dataset_dir = Path(dataset_dir).resolve()
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    target = dataset_dir / "images.csv"
    previous = read_existing(target)
    locations = read_locations(dataset_dir / "locations.csv")
    warnings = set()
    candidates = []
    problems = []

    for image_path in sorted(images_dir.rglob("*"), key=lambda p: p.as_posix().lower()):
        if not image_path.is_file() or any(part.startswith(".") for part in image_path.relative_to(images_dir).parts):
            continue
        if image_path.suffix.lower() not in EXTENSIONS:
            if image_path.suffix.lower() in {".heic", ".heif"}:
                warnings.add(f"Skipped unsupported camera format; convert first: {image_path.name}")
            continue
        if image_path.is_symlink():
            problems.append(f"Image symlinks are not accepted: {image_path}")
            continue
        try:
            image_path.resolve().relative_to(images_dir.resolve())
        except ValueError:
            problems.append(f"Image resolves outside the dataset: {image_path}")
            continue
        parts = image_path.relative_to(images_dir).parts
        if len(parts) != 3 or parts[0] not in CLASS_LABELS:
            problems.append(f"Expected images/<inside|outside>/<group_id>/<image>: {image_path}")
            continue
        class_name, group_id, _ = parts
        prefix = "in" if class_name == "inside" else "out"
        if not re.fullmatch(rf"{prefix}_[0-9]+", group_id):
            problems.append(f"Invalid group folder for {class_name}: {group_id}")
            continue
        location = locations.get(group_id)
        if location and (location.get("class_name") or "").strip() != class_name:
            problems.append(f"locations.csv class conflicts with folder for {group_id}")
            continue
        verified = location and (location.get("boundary_verified") or "").strip().lower() == "yes"
        if not verified:
            if require_verified:
                problems.append(f"Group must be listed and boundary_verified=yes: {group_id}")
                continue
            warnings.add(f"{group_id}: geographic label NOT verified; check before training.")
        relative_path = image_path.relative_to(dataset_dir).as_posix()
        candidates.append((relative_path, class_name, group_id))

    if problems:
        raise ValueError("Scan stopped; images.csv was not changed:\n" + "\n".join(sorted(set(problems))))

    rows = []
    used_ids = set()
    for relative_path, class_name, group_id in candidates:
        old = previous.get(relative_path, {})
        image_id = (old.get("image_id") or "").strip()
        if not image_id:
            stem = re.sub(r"[^a-z0-9_]+", "_", Path(relative_path).stem.lower()).strip("_") or "image"
            digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
            image_id = f"{stem}_{digest}"
        if image_id in used_ids:
            raise ValueError(f"Duplicate image_id {image_id!r}; images.csv was not changed.")
        used_ids.add(image_id)
        row = {
            "image_id": image_id,
            "relative_path": relative_path,
            "label": CLASS_LABELS[class_name],
            "group_id": group_id,
        }
        for field in METADATA_FIELDS:
            row[field] = old.get(field) or ""
        if row["split"] not in {"", "train", "validation", "test"}:
            raise ValueError(f"Invalid existing split for {relative_path}; images.csv was not changed.")
        rows.append(row)

    group_splits = {}
    for row in rows:
        if row["split"]:
            group_splits.setdefault(row["group_id"], set()).add(row["split"])
    conflicts = [group_id for group_id, splits in group_splits.items() if len(splits) > 1]
    if conflicts:
        raise ValueError("Groups have conflicting splits; images.csv was not changed: " + ", ".join(conflicts))

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=dataset_dir,
            prefix=".images_", suffix=".csv.tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    current_paths = {row["relative_path"] for row in rows}
    return {
        "output": target,
        "rows": rows,
        "warnings": sorted(warnings),
        "removed_records": len(set(previous) - current_paths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
        help="Dataset directory; default is ../dataset relative to this script.",
    )
    parser.add_argument(
        "--require-verified", action="store_true",
        help="Refuse to overwrite if any image group is not boundary-verified in locations.csv.",
    )
    args = parser.parse_args()
    try:
        result = scan_dataset(args.dataset_dir, args.require_verified)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Overwritten: {result['output']}")
    for class_name, label in CLASS_LABELS.items():
        class_rows = [row for row in result["rows"] if row["label"] == label]
        groups = {row["group_id"] for row in class_rows}
        print(f"{class_name}: {len(class_rows)} images, {len(groups)} nonempty groups")
    print(f"Total: {len(result['rows'])} images")
    print(f"Removed stale CSV records: {result['removed_records']}")
    print("Existing metadata preserved by relative_path; new metadata fields left blank.")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    print("File discovery does not verify image readability, privacy, geographic truth, or duplicate scenes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())