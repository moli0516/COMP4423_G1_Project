"""Owner A: canonical annotated-pair access."""

from pathlib import Path


class PairDataset:
    def __init__(self, annotation_path: Path, manifest_path: Path) -> None:
        self.annotation_path = annotation_path
        self.manifest_path = manifest_path

    def load(self, split: str) -> list[dict]:
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        raise NotImplementedError("Implement annotation joins and manifest validation")
