"""Filter detections to the basic-scope maximum of one per class."""

from math import isfinite

from deskcheck.domain.detection import Detection


class DetectionFilter:
    """Apply confidence filtering and deterministic per-class selection, not NMS."""

    def __init__(self, confidence_threshold: float) -> None:
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("Confidence threshold must be finite and within [0, 1]")
        self.confidence_threshold = confidence_threshold

    def apply(self, detections: list[Detection]) -> list[Detection]:
        """Return original selected objects sorted by class ID, without mutation.

        Keep confidence >= threshold. Highest confidence wins per class; ties
        use ascending (x1, y1, x2, y2, detection_id). Fully identical keys keep
        the first object. This is a basic-scope cardinality policy, not spatial
        NMS. The detector adapter owns backend NMS; no IoU gate is applied here.

        Raises:
            TypeError: An input is not a Detection or its ID is not a string.
            ValueError: Invalid confidence, class ID or geometry.
        """
        selected: dict[int, Detection] = {}
        for detection in detections:
            self._validate(detection)
            if detection.confidence < self.confidence_threshold:
                continue
            current = selected.get(detection.class_id)
            if current is None or self._rank(detection) < self._rank(current):
                selected[detection.class_id] = detection
        return [selected[class_id] for class_id in sorted(selected)]

    @staticmethod
    def _rank(detection: Detection) -> tuple[float, float, float, float, float, str]:
        box = detection.box
        return (-detection.confidence, box.x1, box.y1, box.x2, box.y2, detection.detection_id)

    @staticmethod
    def _validate(detection: Detection) -> None:
        if not isinstance(detection, Detection):
            raise TypeError("Each input must be a Detection")
        if type(detection.class_id) is not int or detection.class_id < 0:
            raise ValueError("class_id must be a nonnegative integer")
        if not isinstance(detection.detection_id, str):
            raise TypeError("detection_id must be a string")
        confidence = detection.confidence
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Detection confidence must be finite and within [0, 1]")
        box = detection.box
        if not all(isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2)):
            raise ValueError("Box coordinates must be finite")
        if not (0 <= box.x1 < box.x2 and 0 <= box.y1 < box.y2):
            raise ValueError("Box must have positive area and nonnegative coordinates")
