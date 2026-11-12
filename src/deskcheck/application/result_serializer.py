"""Strict JSON/CSV transport of B events; no second change classifier."""

import csv
import io
import json
from math import isfinite

from deskcheck.application.app_config import CLASS_NAMES
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.evaluation.event_evaluator import EventEvaluator


def _string_keys(value) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _string_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _string_keys(item)


class ResultSerializer:
    def snapshot(self, result: dict) -> dict:
        _string_keys(result)
        data = json.loads(json.dumps(result, allow_nan=False))
        if not isinstance(data, dict):
            raise TypeError("Result must be an object")
        pair_id = data.get("pair_id")
        size = data.get("image_size")
        records = data.get("records")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError("Result requires a nonempty pair_id")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
        ):
            raise ValueError("Result requires [width, height]")
        if not isinstance(records, list):
            raise TypeError("Result requires an explicit records list")
        EventEvaluator().evaluate({pair_id: records}, {pair_id: records})
        required = {
            "class_id",
            "status",
            "before_detection_id",
            "after_detection_id",
            "before_box",
            "after_box",
            "distance_norm",
            "movement_threshold",
            "indicates_change",
        }
        for record in records:
            if not required <= record.keys():
                raise ValueError("Result record is missing B evidence fields")
            for side in ("before", "after"):
                box = record[f"{side}_box"]
                if box is not None:
                    BoundingBox(*box).validate(*size)
        if data.get("schema_version", "1.0") != "1.0":
            raise ValueError("Unsupported result schema version")
        # Additive result extension; older event-only fixtures remain supported.
        sides = ("before_detections", "after_detections")
        if any(side in data for side in sides):
            if not all(side in data for side in sides):
                raise ValueError("Retained detections must include both sides")
            for side in sides:
                selected = data[side]
                if not isinstance(selected, list):
                    raise TypeError("Retained detections must be a list")
                used_classes = set()
                used_ids = set()
                for detection in selected:
                    if not isinstance(detection, dict) or set(detection) != {
                        "detection_id",
                        "class_id",
                        "confidence",
                        "bbox_xyxy",
                    }:
                        raise ValueError("Retained detection fields are invalid")
                    name = detection["detection_id"]
                    class_id = detection["class_id"]
                    confidence = detection["confidence"]
                    box = detection["bbox_xyxy"]
                    if not isinstance(name, str) or not name.strip() or name in used_ids:
                        raise ValueError("Retained detection IDs must be unique strings")
                    if type(class_id) is not int or not 0 <= class_id < len(CLASS_NAMES):
                        raise ValueError("Retained detection class is invalid")
                    if class_id in used_classes:
                        raise ValueError("Only one B-filtered detection per class is permitted")
                    if (
                        type(confidence) not in (int, float)
                        or not isfinite(confidence)
                        or not 0 <= confidence <= 1
                    ):
                        raise ValueError("Retained detection confidence is invalid")
                    if (
                        not isinstance(box, list)
                        or len(box) != 4
                        or any(
                            type(coord) not in (int, float) or not isfinite(coord) for coord in box
                        )
                    ):
                        raise ValueError(
                            "Retained bounding box must contain four finite coordinates"
                        )
                    BoundingBox(*box).validate(*size)
                    used_ids.add(name)
                    used_classes.add(class_id)
                # Link evidence, without recomputing B's matching/status decisions.
                prefix = side.removesuffix("_detections")
                by_class = {d["class_id"]: d for d in selected}
                expected_classes = {
                    r["class_id"] for r in records if r[f"{prefix}_box"] is not None
                }
                if by_class.keys() != expected_classes:
                    raise ValueError("Retained detection inventory disagrees with event records")
                for record in records:
                    detection = by_class.get(record["class_id"])
                    expected_id = detection["detection_id"] if detection is not None else None
                    expected_box = detection["bbox_xyxy"] if detection is not None else None
                    if (
                        record[f"{prefix}_detection_id"] != expected_id
                        or record[f"{prefix}_box"] != expected_box
                    ):
                        raise ValueError("Retained detection evidence disagrees with event record")
        return data

    def to_json(self, result: dict) -> str:
        return json.dumps(self.snapshot(result), indent=2, sort_keys=True, allow_nan=False) + "\n"

    def to_csv(self, result: dict) -> str:
        data = self.snapshot(result)
        columns = ["pair_id", "class_id", "status", "before_detection_id", "after_detection_id"]
        columns += [
            f"{side}_{coord}" for side in ("before", "after") for coord in ("x1", "y1", "x2", "y2")
        ]
        columns += ["distance_norm", "movement_threshold", "indicates_change"]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for record in data["records"]:
            row = {key: record.get(key) for key in columns}
            row["pair_id"] = data["pair_id"]
            for side in ("before", "after"):
                for key, value in zip(
                    ("x1", "y1", "x2", "y2"), record[f"{side}_box"] or [None] * 4
                ):
                    row[f"{side}_{key}"] = value
            writer.writerow(row)
        return output.getvalue()
