"""Owner A/C: an in-memory decoded image. Pure data, no I/O.

Ownership policy (from B's review, 10 Nov 2026):

- `array` is canonical, orientation-normalized RGB uint8 data with shape
  (H, W, 3). No model-specific resizing, cropping, or letterboxing.
- `width` and `height` are positive and must match `array.shape`.
- `source_path` is provenance only. Consumers (especially the detector) must
  work with `source_path=None` and must NOT reopen the file.
- `frozen=True` prevents reassigning attributes but does NOT make the numpy
  buffer immutable. Callers must not mutate `array` in place. Consumers that
  need to modify pixels (rendering, augmentation preview) must call `.copy()`
  and use the returned ImageFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageFrame:
    array: np.ndarray          # RGB uint8, shape (H, W, 3)
    width: int
    height: int
    source_path: Path | None = None   # provenance only, never a feature

    def __post_init__(self) -> None:
        if not isinstance(self.array, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")
        if self.array.dtype != np.uint8:
            raise ValueError("array dtype must be uint8")
        if self.array.ndim != 3 or self.array.shape[2] != 3:
            raise ValueError("array shape must be (H, W, 3) RGB")
        h, w, _ = self.array.shape
        if h != self.height or w != self.width:
            raise ValueError(
                f"array shape {(h, w)} disagrees with "
                f"(height,width)=({self.height},{self.width})"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    def copy(self) -> ImageFrame:
        """Return a new ImageFrame with a deep-copied array.

        Use this before in-place modification (rendering, augmentation
        previews). The original array is never returned to callers that
        intend to mutate.
        """
        return ImageFrame(
            array=self.array.copy(),
            width=self.width,
            height=self.height,
            source_path=self.source_path,
        )
