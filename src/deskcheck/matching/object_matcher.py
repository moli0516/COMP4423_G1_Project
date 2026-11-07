"""Associate filtered detections by class, never by spatial distance."""

from deskcheck.domain.detection import Detection


class ObjectMatcher:
    """Match at most one detection per class on each side of a pair."""

    def match(
        self, before: list[Detection], after: list[Detection]
    ) -> list[tuple[Detection | None, Detection | None]]:
        """Return matches in ascending class-ID order without modifying inputs.

        A class present on only one side has None on the other side. Classes
        absent from both sides produce no entry. Confidence and geometry do
        not affect association: callers must filter detections first.

        Raises:
            ValueError: Either input contains duplicate class IDs.
        """
        before_by_class = self._index_by_class(before, side="before")
        after_by_class = self._index_by_class(after, side="after")
        class_ids = sorted(before_by_class.keys() | after_by_class.keys())
        return [
            (before_by_class.get(class_id), after_by_class.get(class_id))
            for class_id in class_ids
        ]

    @staticmethod
    def _index_by_class(detections: list[Detection], *, side: str) -> dict[int, Detection]:
        indexed: dict[int, Detection] = {}
        for detection in detections:
            if detection.class_id in indexed:
                raise ValueError(
                    f"Duplicate class_id {detection.class_id} in {side} detections; "
                    "filter to at most one detection per class before matching."
                )
            indexed[detection.class_id] = detection
        return indexed
