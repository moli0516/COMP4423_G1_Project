"""Canonical detector output in original-image pixels."""

from dataclasses import dataclass

from deskcheck.domain.bounding_box import BoundingBox


@dataclass(frozen=True)
class Detection:
    detection_id: str
    class_id: int
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between zero and one")
        if self.class_id < 0:
            raise ValueError("Class ID must be nonnegative")
