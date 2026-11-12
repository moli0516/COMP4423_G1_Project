"""Explicit C-to-A frame conversion seam; not a trained detector implementation."""

from collections.abc import Callable
from typing import Any

import numpy as np

from deskcheck.application.image_frame import ImageFrame
from deskcheck.domain.detection import Detection


class FrameDetectorAdapter:
    """Wrap a supplied detector and a supplied shared-frame constructor.

    Does not import or overwrite A's unmerged interface. The factory must accept
    array, width, height and source_path. It receives a copy and no path identity.
    Model preprocessing, canonical box restoration and backend NMS remain A's job.
    This seam does not enable real mode or establish model accuracy.
    """

    def __init__(self, detector: Any, frame_factory: Callable[..., Any]) -> None:
        if not callable(getattr(detector, "predict", None)) or not callable(frame_factory):
            raise TypeError("A callable detector.predict and frame_factory are required")
        self._detector = detector
        self._factory = frame_factory

    def predict(self, frame: ImageFrame) -> list[Detection]:
        if not isinstance(frame, ImageFrame):
            raise TypeError("Expected a C-local ImageFrame")
        converted = self._factory(
            array=frame.array.copy(), width=frame.width, height=frame.height, source_path=None
        )
        self._validate(converted, frame, "Frame factory changed canonical pixels or dimensions")
        result = self._detector.predict(converted)
        self._validate(converted, frame, "Detector mutated canonical input pixels or dimensions")
        if not isinstance(result, list) or any(not isinstance(d, Detection) for d in result):
            raise TypeError("Detector must return a list of shared Detection objects")
        return result

    @staticmethod
    def _validate(converted: Any, frame: ImageFrame, message: str) -> None:
        pixels = getattr(converted, "array", None)
        if (
            not isinstance(pixels, np.ndarray)
            or pixels.dtype != np.uint8
            or pixels.shape != frame.array.shape
            or getattr(converted, "width", None) != frame.width
            or getattr(converted, "height", None) != frame.height
            or getattr(converted, "source_path", None) is not None
            or not np.array_equal(pixels, frame.array)
        ):
            raise ValueError(message)
