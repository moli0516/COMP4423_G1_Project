#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image

CLASSES = ["bottle", "cup", "keyboard", "mouse", "book", "scissors"]
ALL_CLASSES = set(range(len(CLASSES)))

EXPECTED_PAIRS = {
    "P1": ("p1_smallMove1.png", "p1_smallMove2.png"),
    "P2": ("p2_midMove1.png", "p2_midMove2.png"),
    "P3": ("p3_bigMove1.png", "p3_bigMove2.png"),
    "P4": ("p4_sub1.png", "p4_sub2.png"),
    "P5": ("p5_add1.png", "p5_add2.png"),
    "P6": ("p6_control1.png", "p6_control2.png"),
}

MOVED_CLASS = {"P1": 0, "P2": 1, "P3": 5}
ABSENT_BEFORE = {"P5": {4}}
ABSENT_AFTER = {"P4": {3}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_labels(path):
    require(path.is_file(), f"Missing label file: {path}")
    result = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue

        fields = line.split()
        require(
            len(fields) == 5,
            f"{path}:{line_number}: expected 5 YOLO fields",
        )

        try:
            class_id = int(fields[0])
            cx, cy, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid YOLO values"
            ) from exc

        require(class_id in ALL_CLASSES, f"{path}: invalid class {class_id}")
        require(class_id not in result, f"{path}: duplicate class {class_id}")
        require(
            all(math.isfinite(v) for v in (cx, cy, width, height)),
            f"{path}: non-finite coordinates",
        )
        require(
            0 <= cx <= 1 and 0 <= cy <= 1
            and 0 < width <= 1 and 0 < height <= 1,
            f"{path}: invalid normalized box for class {class_id}",
        )

        tolerance = 1e-5
        require(
            cx - width / 2 >= -tolerance
            and cy - height / 2 >= -tolerance
            and cx + width / 2 <= 1 + tolerance
            and cy + height / 2 <= 1 + tolerance,
            f"{path}: box outside image for class {class_id}",
        )

        result[class_id] = (cx, cy, width, height)

    return result


def inspect_image(path):
    require(path.is_file(), f"Missing image: {path}")
    with Image.open(path) as image:
        image.load()
        require(image.mode == "RGB", f"{path}: expected RGB, got {image.mode}")
        require(
            image.getexif().get(274, 1) == 1,
            f"{path}: unexpected EXIF orientation",
        )
        return image.size


def make_parser():
    parser = argparse.ArgumentParser(
        description="Measure manual-box pilot distances and propose tau."
    )
    parser.add_argument(
        "--capture-csv",
        type=Path,
        default=Path("data/raw/pilot_photos/pilot_capture_records.csv"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/obj_train_data"),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path(
            "data/annotations/pilot_tau_20261112/yolo/obj_train_data"
        ),
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=Path("data/annotations/pilot_tau_20261112/yolo/obj.names"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pilot_tau_20261112"),
    )
    parser.add_argument(
        "--accept-small-move",
        action="store_true",
        help="Confirm P1 bottle movement is intended to count as moved.",
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if not args.accept_small_move:
        parser.error(
            "Confirm that P1's small bottle movement should count as moved "
            "by supplying --accept-small-move."
        )

    input_hashes = {}

    def record_input(path):
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    require(args.names.is_file(), f"Missing class names: {args.names}")
    names = [
        line.strip()
        for line in args.names.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    require(names == CLASSES, f"Unexpected class mapping: {names}")
    record_input(args.names)

    capture_bytes = args.capture_csv.read_bytes()
    record_input(args.capture_csv)
    captures = list(
        csv.DictReader(io.StringIO(capture_bytes.decode("utf-8-sig")))
    )

    required_fields = {
        "pair_id",
        "canonical_before",
        "canonical_after",
        "capture_group_id",
        "session",
        "change_description",
        "camera_changed",
        "other_objects_touched",
    }

    require(len(captures) == 6, "Expected exactly 6 capture rows")
    require(
        all(required_fields <= set(row) for row in captures),
        "Capture CSV is missing required columns",
    )
    pair_ids = [row["pair_id"].strip() for row in captures]
    require(
        len(set(pair_ids)) == 6 and set(pair_ids) == set(EXPECTED_PAIRS),
        f"Unexpected or duplicate pair IDs: {pair_ids}",
    )

    measurements = []

    for capture in sorted(captures, key=lambda row: row["pair_id"]):
        pair_id = capture["pair_id"].strip()
        before_name = capture["canonical_before"].strip()
        after_name = capture["canonical_after"].strip()

        require(
            (before_name, after_name) == EXPECTED_PAIRS[pair_id],
            f"{pair_id}: unexpected before/after filenames",
        )
        require(
            capture["capture_group_id"].strip() == "pilot_g01",
            f"{pair_id}: unexpected capture group",
        )
        require(
            capture["session"].strip().lower() == "single continuous session",
            f"{pair_id}: unexpected session",
        )
        for field in ("camera_changed", "other_objects_touched"):
            require(
                capture[field].strip().lower() == "no",
                f"{pair_id}: {field} must be 'no'; review classification",
            )

        before_image = args.image_dir / before_name
        after_image = args.image_dir / after_name
        before_label = args.label_dir / Path(before_name).with_suffix(".txt")
        after_label = args.label_dir / Path(after_name).with_suffix(".txt")

        before_size = inspect_image(before_image)
        after_size = inspect_image(after_image)
        require(
            before_size == after_size,
            f"{pair_id}: before/after dimensions differ",
        )
        require(
            before_size == (2048, 1536),
            f"{pair_id}: expected 2048x1536, got {before_size}",
        )

        before = read_labels(before_label)
        after = read_labels(after_label)

        require(
            set(before) == ALL_CLASSES - ABSENT_BEFORE.get(pair_id, set()),
            f"{pair_id}: unexpected before class inventory: {sorted(before)}",
        )
        require(
            set(after) == ALL_CLASSES - ABSENT_AFTER.get(pair_id, set()),
            f"{pair_id}: unexpected after class inventory: {sorted(after)}",
        )

        for path in (before_image, after_image, before_label, after_label):
            record_input(path)

        width, height = before_size
        diagonal = math.hypot(width, height)

        for class_id in sorted(set(before) | set(after)):
            before_box = before.get(class_id)
            after_box = after.get(class_id)

            row = {
                "pair_id": pair_id,
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "before_filename": before_name,
                "after_filename": after_name,
                "before_present": before_box is not None,
                "after_present": after_box is not None,
                "image_width": width,
                "image_height": height,
                "operation_label": "",
                "include_for_tau": False,
                "before_cx_px": None,
                "before_cy_px": None,
                "after_cx_px": None,
                "after_cy_px": None,
                "displacement_px": None,
                "normalized_distance": None,
                "capture_description": capture["change_description"],
            }

            if before_box is not None:
                row["before_cx_px"] = before_box[0] * width
                row["before_cy_px"] = before_box[1] * height

            if after_box is not None:
                row["after_cx_px"] = after_box[0] * width
                row["after_cy_px"] = after_box[1] * height

            if before_box is None:
                row["operation_label"] = "added"
            elif after_box is None:
                row["operation_label"] = "removed"
            else:
                row["operation_label"] = (
                    "reportable_movement"
                    if MOVED_CLASS.get(pair_id) == class_id
                    else "untouched"
                )
                row["include_for_tau"] = True

                dx = row["after_cx_px"] - row["before_cx_px"]
                dy = row["after_cy_px"] - row["before_cy_px"]
                row["displacement_px"] = math.hypot(dx, dy)
                row["normalized_distance"] = (
                    row["displacement_px"] / diagonal
                )

            measurements.append(row)

    untouched = [
        row for row in measurements
        if row["operation_label"] == "untouched"
    ]
    moved = [
        row for row in measurements
        if row["operation_label"] == "reportable_movement"
    ]
    added = [
        row for row in measurements if row["operation_label"] == "added"
    ]
    removed = [
        row for row in measurements if row["operation_label"] == "removed"
    ]

    require(
        (len(untouched), len(moved), len(added), len(removed)) == (31, 3, 1, 1),
        "Unexpected operation counts",
    )

    upper_untouched = max(
        row["normalized_distance"] for row in untouched
    )
    lower_moved = min(
        row["normalized_distance"] for row in moved
    )
    separable = upper_untouched < lower_moved
    tau = (
        (upper_untouched + lower_moved) / 2
        if separable else None
    )

    if separable:
        require(
            upper_untouched <= tau < lower_moved,
            "Midpoint is not representable inside the valid interval",
        )

    for row in measurements:
        distance = row["normalized_distance"]
        row["status_at_candidate_tau"] = (
            row["operation_label"]
            if distance is None
            else (
                "not_evaluated"
                if tau is None
                else ("moved" if distance > tau else "unchanged")
            )
        )

    report = {
        "status": (
            "candidate_requires_review"
            if separable else "no_separating_threshold"
        ),
        "basis": "manual YOLO boxes, not detector predictions",
        "decision_rule": "moved iff normalized_distance > tau",
        "normalization": "Euclidean center displacement / image diagonal",
        "small_move_accepted_as_reportable": True,
        "counts": {
            "untouched": len(untouched),
            "reportable_movement": len(moved),
            "added": len(added),
            "removed": len(removed),
        },
        "U_max_untouched": upper_untouched,
        "M_min_reportable_movement": lower_moved,
        "separation_gap_M_minus_U": lower_moved - upper_untouched,
        "valid_interval": (
            {"lower_inclusive": upper_untouched,
             "upper_exclusive": lower_moved}
            if separable else None
        ),
        "candidate_tau": tau,
        "selection_rule": "midpoint of U and M if U < M; otherwise no candidate",
        "max_untouched_records": [
            row for row in untouched
            if row["normalized_distance"] == upper_untouched
        ],
        "min_movement_records": [
            row for row in moved
            if row["normalized_distance"] == lower_moved
        ],
        "input_sha256": input_hashes,
        "limitations": [
            "Overlay/annotation review must be completed separately.",
            "Six pairs from one continuous capture session.",
            "No detector-localization or real-checkpoint validation.",
            "Candidate is not automatically frozen or written to application config.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pilot_distances.csv"
    report_path = args.output_dir / "tau_report.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Untouched: {len(untouched)}; movement: {len(moved)}")
    print(f"U = {upper_untouched:.12g}")
    print(f"M = {lower_moved:.12g}")
    if separable:
        print(f"Valid interval: {upper_untouched:.12g} <= tau < {lower_moved:.12g}")
        print(f"Candidate tau: {tau:.12g}")
        print("Candidate only: review overlays and report before freezing.")
    else:
        print("No separating threshold: inspect labels and movement definition.")
    print(f"Measurements: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()