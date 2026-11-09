from pathlib import Path

from src.exceptions import TrainingError
from src.utils.file_io import read_table, resolve_image_path, sha256_file
from .constants import (
    CLASS_NAMES, SPLIT_NAMES, REQUIRED_IMAGE_COLUMNS, REQUIRED_LOCATION_COLUMNS,
)


class DatasetCatalog:
    """Read and validate metadata without decoding images."""

    def __init__(self, dataset_dir, allow_unverified=False):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.allow_unverified = allow_unverified

    def load(self):
        rows = read_table(self.dataset_dir / "images.csv", REQUIRED_IMAGE_COLUMNS)
        locations = self._read_locations()
        if not rows:
            raise TrainingError("images.csv has no image rows.")
        groups, paths, identifiers = {}, set(), set()
        for row in rows:
            self._validate_image(row, groups, paths, identifiers)
        for group_id, (split, label) in groups.items():
            location = locations.get(group_id)
            if location is None:
                raise TrainingError(f"Group {group_id} is missing from locations.csv.")
            if location["class_name"] != CLASS_NAMES[int(label)]:
                raise TrainingError(f"Group {group_id}: class conflicts with image labels.")
            if not self.allow_unverified and location["boundary_verified"].lower() != "yes":
                raise TrainingError(
                    f"Group {group_id} is not boundary-verified; "
                    "use --allow-unverified only for development runs."
                )
        by_split = {name: [r for r in rows if r["split"] == name] for name in SPLIT_NAMES}
        for name, subset in by_split.items():
            if {row["label"] for row in subset} != {"0", "1"}:
                raise TrainingError(f"Split {name} must contain both classes.")
        return {
            "dataset_dir": self.dataset_dir, "rows": rows, "by_split": by_split,
            "images_csv_sha256": sha256_file(self.dataset_dir / "images.csv"),
        }

    def _read_locations(self):
        locations = {}
        for row in read_table(self.dataset_dir / "locations.csv", REQUIRED_LOCATION_COLUMNS):
            group_id = row["group_id"]
            if not group_id or group_id in locations:
                raise TrainingError(f"locations.csv: empty or duplicate group_id {group_id!r}.")
            locations[group_id] = row
        return locations

    def _validate_image(self, row, groups, paths, identifiers):
        place = f"images.csv:{row['_line']}"
        for name in REQUIRED_IMAGE_COLUMNS:
            if not row[name]:
                raise TrainingError((f"{place}: empty split. Run scripts/split_dataset.py first." if name == "split" else f"{place}: empty {name}."))
        if row["label"] not in {"0", "1"}:
            raise TrainingError(f"{place}: label must be 0 or 1.")
        if row["split"] not in SPLIT_NAMES:
            raise TrainingError(f"{place}: split must be train, validation, or test.")
        if row["image_id"] in identifiers:
            raise TrainingError(f"{place}: duplicate image_id {row['image_id']}.")
        path = resolve_image_path(self.dataset_dir, row["relative_path"])
        if path in paths:
            raise TrainingError(f"{place}: duplicate relative_path {row['relative_path']}.")
        if not path.is_file():
            raise TrainingError(f"{place}: file does not exist: {row['relative_path']}")
        identifiers.add(row["image_id"])
        paths.add(path)
        pair = (row["split"], row["label"])
        previous = groups.setdefault(row["group_id"], pair)
        if previous[0] != pair[0]:
            raise TrainingError(f"Group {row['group_id']} appears in more than one split.")
        if previous[1] != pair[1]:
            raise TrainingError(f"Group {row['group_id']} contains both labels.")
