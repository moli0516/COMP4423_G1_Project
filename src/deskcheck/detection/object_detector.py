"""Backend-independent detector interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from deskcheck.domain.detection import Detection


class ObjectDetector(ABC):
    @abstractmethod
    def predict(self, image_path: Path) -> list[Detection]:
        """Use image content only; return original-image pixel coordinates.

        The path locates an image, never encodes a prediction feature.
        The adapter must apply consistent orientation/RGB preprocessing.
        """
        raise NotImplementedError
