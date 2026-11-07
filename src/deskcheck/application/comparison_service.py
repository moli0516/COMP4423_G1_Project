"""Owner C: the single inference orchestration entry point."""

from pathlib import Path


class ComparisonService:
    def compare(self, before_path: Path, after_path: Path, pair_id: str) -> dict:
        raise NotImplementedError("Integrate loader, trained detector, filter, matcher and analyzer")
