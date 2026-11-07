"""Classify already-matched detections using explicit presence/movement rules."""

from math import hypot, isfinite

from deskcheck.domain.change_status import ChangeStatus
from deskcheck.domain.detection import Detection


class ChangeAnalyzer:
    """Use a caller-approved threshold; no default scientific threshold is supplied."""

    def __init__(self, movement_threshold: float) -> None:
        if (
            isinstance(movement_threshold, bool)
            or not isinstance(movement_threshold, (int, float))
            or not isfinite(movement_threshold)
            or not 0 <= movement_threshold <= 1
        ):
            raise ValueError("Movement threshold must be finite and within [0, 1]")
        self.movement_threshold = movement_threshold

    def analyze(
        self,
        matches: list[tuple[Detection | None, Detection | None]],
        image_size: tuple[int, int],
    ) -> list[dict]:
        """Return JSON-safe event dictionaries in ascending class-ID order.

        Image size is (width, height) of both original oriented images. Boxes
        must already be in that coordinate system. Matched sides must have
        equal class IDs. No spatial association gate or confidence filter is
        applied here. Pair ID and artifact/rules provenance belong to the caller.

        Raises:
            ValueError: Invalid dimensions, boxes, pairs or duplicate classes.
        """
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size must be a (width, height) tuple")
        width, height = image_size
        if any(type(value) is not int or value <= 0 for value in image_size):
            raise ValueError("Image dimensions must be positive integers")
        try:
            diagonal = hypot(width, height)
        except OverflowError as exc:
            raise ValueError("Image dimensions are too large") from exc
        if not isfinite(diagonal):
            raise ValueError("Image diagonal must be finite")

        records: list[dict] = []
        seen_classes: set[int] = set()
        for match in matches:
            if not isinstance(match, (tuple, list)) or len(match) != 2:
                raise ValueError("Each match must contain exactly two sides")
            before, after = match
            if before is None and after is None:
                raise ValueError("A match must contain at least one detection")
            for detection in (before, after):
                if detection is not None:
                    if not isinstance(detection, Detection):
                        raise ValueError("Match sides must be Detection objects or None")
                    detection.box.validate(width, height)
            if before is not None and after is not None and before.class_id != after.class_id:
                raise ValueError("Matched detections must have the same class_id")
            present = before if before is not None else after
            assert present is not None
            class_id = present.class_id
            if class_id in seen_classes:
                raise ValueError(f"Duplicate class_id {class_id} across matches")
            seen_classes.add(class_id)

            distance: float | None = None
            if before is None:
                status = ChangeStatus.ADDED
            elif after is None:
                status = ChangeStatus.REMOVED
            else:
                before_x, before_y = before.box.center()
                after_x, after_y = after.box.center()
                distance = hypot(after_x - before_x, after_y - before_y) / diagonal
                status = (
                    ChangeStatus.MOVED
                    if distance > self.movement_threshold
                    else ChangeStatus.UNCHANGED
                )
            records.append(
                {
                    "class_id": class_id,
                    "status": status.value,
                    "before_detection_id": before.detection_id if before is not None else None,
                    "after_detection_id": after.detection_id if after is not None else None,
                    "before_box": self._box_values(before),
                    "after_box": self._box_values(after),
                    "distance_norm": distance,
                    "movement_threshold": self.movement_threshold,
                    "indicates_change": status.indicates_change,
                }
            )
        return sorted(records, key=lambda record: record["class_id"])

    @staticmethod
    def _box_values(detection: Detection | None) -> list[float] | None:
        if detection is None:
            return None
        box = detection.box
        return [box.x1, box.y1, box.x2, box.y2]
