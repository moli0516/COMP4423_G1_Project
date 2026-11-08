"""Basic-scope, localized event scoring; not detector AP evaluation."""

from math import isfinite


class EventEvaluator:
    """Score one event per class per pair using status-independent localization."""

    STATUSES = ("added", "removed", "moved", "unchanged")
    CHANGE_STATUSES = ("added", "removed", "moved")

    def __init__(self, iou_threshold: float = 0.5) -> None:
        if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
            raise TypeError("IoU threshold must be numeric")
        if not isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("IoU threshold must be finite and in (0, 1]")
        self.iou_threshold = iou_threshold

    def evaluate(self, predicted: dict, ground_truth: dict) -> dict:
        """Accept {pair_id: [event, ...]} with identical nonempty pair inventories.

        Event fields: class_id, status, before_box, after_box. Additional fields
        are ignored. Empty scene results must be explicit empty lists. No missing
        pair is silently treated as an empty prediction. Scope: one event/class/pair.
        """
        self._validate_mapping(predicted)
        self._validate_mapping(ground_truth)
        if not ground_truth:
            raise ValueError("Evaluation requires at least one pair")
        if predicted.keys() != ground_truth.keys():
            raise ValueError("Prediction and ground-truth pair IDs must match exactly")
        counts = {status: {"tp": 0, "fp": 0, "fn": 0} for status in self.STATUSES}
        for pair_id in sorted(ground_truth):
            predictions = self._index(predicted[pair_id])
            truths = self._index(ground_truth[pair_id])
            for class_id in sorted(predictions.keys() | truths.keys()):
                prediction, truth = predictions.get(class_id), truths.get(class_id)
                if prediction is not None and truth is not None:
                    correct = (
                        prediction["status"] == truth["status"]
                        and self._localized(prediction, truth)
                    )
                    if correct:
                        counts[truth["status"]]["tp"] += 1
                    else:
                        counts[prediction["status"]]["fp"] += 1
                        counts[truth["status"]]["fn"] += 1
                elif prediction is not None:
                    counts[prediction["status"]]["fp"] += 1
                elif truth is not None:
                    counts[truth["status"]]["fn"] += 1
        per_status = {}
        for status, values in counts.items():
            tp, fp, fn = values["tp"], values["fp"], values["fn"]
            per_status[status] = {
                **values,
                "support": tp + fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            }
        return {
            "protocol_version": "basic-localized-events-v1",
            "pair_count": len(ground_truth),
            "iou_threshold": self.iou_threshold,
            "zero_division": 0,
            "per_status": per_status,
            "macro_f1": sum(per_status[s]["f1"] for s in self.STATUSES) / 4,
            "change_only_macro_f1": sum(
                per_status[s]["f1"] for s in self.CHANGE_STATUSES
            ) / 3,
            "averaging": "fixed status set; absent-status F1 is zero",
        }

    @staticmethod
    def _validate_mapping(value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("Events must be a dictionary keyed by pair ID")
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("Pair IDs must be nonempty strings")

    @classmethod
    def _index(cls, events: list) -> dict:
        if not isinstance(events, list):
            raise TypeError("Each pair must contain a list of events")
        indexed = {}
        for event in events:
            if not isinstance(event, dict):
                raise TypeError("Each event must be a dictionary")
            required = {"class_id", "status", "before_box", "after_box"}
            if not required <= event.keys():
                raise ValueError("Event is missing required fields")
            class_id = event["class_id"]
            if type(class_id) is not int or class_id < 0:
                raise ValueError("class_id must be a nonnegative integer")
            if class_id in indexed:
                raise ValueError("Duplicate class_id within a pair")
            status = event["status"]
            if status not in cls.STATUSES:
                raise ValueError("Unknown event status")
            before, after = event["before_box"], event["after_box"]
            expected_presence = {
                "added": (False, True), "removed": (True, False),
                "moved": (True, True), "unchanged": (True, True),
            }[status]
            if (before is not None, after is not None) != expected_presence:
                raise ValueError("Status and box presence disagree")
            for box in (before, after):
                if box is not None:
                    cls._validate_box(box)
            indexed[class_id] = event
        return indexed

    @staticmethod
    def _validate_box(box: list) -> None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Box must have four xyxy coordinates")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in box
        ):
            raise ValueError("Box coordinates must be finite numbers")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("Box must have nonnegative coordinates and positive area")

    def _localized(self, prediction: dict, truth: dict) -> bool:
        shared = [side for side in ("before_box", "after_box")
                  if prediction[side] is not None and truth[side] is not None]
        return bool(shared) and all(
            self._iou(prediction[side], truth[side]) >= self.iou_threshold for side in shared
        )

    @staticmethod
    def _iou(a: list, b: list) -> float:
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
        return intersection / union
