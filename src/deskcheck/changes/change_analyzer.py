"""Owner B: movement and event classification."""

from math import isfinite


class ChangeAnalyzer:
    def __init__(self, movement_threshold: float) -> None:
        if not isfinite(movement_threshold) or not 0 <= movement_threshold <= 1:
            raise ValueError("Movement threshold must be finite and within [0, 1]")
        self.movement_threshold = movement_threshold

    def analyze(self, matches: list, image_size: tuple[int, int]) -> list[dict]:
        raise NotImplementedError("Implement presence-first rules and normalized displacement")
