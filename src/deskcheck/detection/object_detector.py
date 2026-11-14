"""Backend-independent detector interface.

Contract (from B, 10 Nov 2026):

- predict() takes a canonical ImageFrame (RGB uint8, no model resize applied).
- predict() returns Detection objects in ImageFrame pixel coordinates.
- The adapter performs model-specific resize/letterbox internally and undoes
  it before returning, so returned boxes are in ImageFrame coordinates.
- The adapter must work with frame.source_path=None. It must never reopen
  the file; all pixel input comes from frame.array.
- The adapter must not mutate frame.array in place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from deskcheck.domain.detection import Detection
from deskcheck.domain.image_frame import ImageFrame


class ObjectDetector(ABC):
    @abstractmethod
    def predict(self, image: ImageFrame) -> list[Detection]:
        """Return detections in ImageFrame pixel coordinates.

        Rules:
        - Use only image.array as input; source_path is provenance.
        - Do not mutate image.array.
        - Return canonical class_ids from the approved class map.
        - Apply backend NMS before returning (caller does not).
        - Undo any model-specific resize/letterboxing.
        """
        raise NotImplementedError
