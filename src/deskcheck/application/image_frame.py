"""C-local pixel boundary pending the shared A/B ImageFrame contract."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageFrame:
    array: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.array, np.ndarray)
            or self.array.dtype != np.uint8
            or self.array.ndim != 3
            or self.array.shape[2] != 3
            or min(self.array.shape[:2]) <= 0
        ):
            raise ValueError("ImageFrame requires nonempty RGB uint8 HxWx3 pixels")
        pixels = np.frombuffer(self.array.tobytes(order="C"), dtype=np.uint8).reshape(
            self.array.shape
        )
        object.__setattr__(self, "array", pixels)

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def height(self) -> int:
        return self.array.shape[0]
