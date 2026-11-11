"""Strict JSON/CSV transport of B events; no second change classifier."""

import csv
import io
import json

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
