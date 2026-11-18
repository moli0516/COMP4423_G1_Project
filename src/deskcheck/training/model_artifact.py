"""Owner A: the metadata record that accompanies a trained checkpoint.

A ModelArtifact captures everything needed to reproduce or audit a training
run: the checkpoint file, its hash, the resolved config, the dataset
identity, the selected epoch, and the environment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    """A trained checkpoint plus everything needed to audit it."""

    # Identity
    experiment_id: str
    run_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str

    # Model
    architecture: str
    pretrained_weights: str
    class_names: tuple[str, ...]
    image_size: int

    # Data
    dataset_yaml: str
    dataset_release: str
    split_seed: int
    training_manifest_sha256: str

    # Training
    epochs_requested: int
    epochs_completed: int
    selected_epoch: int
    checkpoint_selection: str
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    seed: int
    device: str

    # Inference defaults (frozen for evaluation)
    confidence_threshold: float
    nms_iou_threshold: float

    # Provenance
    ultralytics_version: str
    torch_version: str
    python_version: str
    resolved_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable dict. Paths become strings."""
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        d["checkpoint_path"] = str(self.checkpoint_path)
        d["class_names"] = list(self.class_names)
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ModelArtifact:
        d = json.loads(path.read_text(encoding="utf-8"))
        d["run_dir"] = Path(d["run_dir"])
        d["checkpoint_path"] = Path(d["checkpoint_path"])
        d["class_names"] = tuple(d["class_names"])
        return cls(**d)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
