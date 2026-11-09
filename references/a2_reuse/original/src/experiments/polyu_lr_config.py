import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolyuLRConfig:
    dataset_dir: Path
    reference_metadata: Path
    reference_report: Path
    output_dir: Path
    c_values: tuple = (0.01, 0.1, 1.0)
    solver: str = "liblinear"
    max_iter: int = 10000
    tol: float = 1e-4
    seed: int = 42
    allow_unverified: bool = True
    effective_zero_threshold: float = 1e-8

    def __post_init__(self):
        for name in ("dataset_dir", "reference_metadata", "reference_report", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if not isinstance(self.c_values, (list, tuple)) or any(type(value) not in (int, float) for value in self.c_values):
            raise ValueError("c_values must contain numeric values, not strings or booleans.")
        object.__setattr__(self, "c_values", tuple(float(value) for value in self.c_values))
        if self.c_values != (0.01, 0.1, 1.0):
            raise ValueError("This frozen experiment requires C values [0.01, 0.1, 1.0].")
        if self.solver != "liblinear":
            raise ValueError("Keep liblinear fixed for this L1/L2 comparison.")
        if type(self.max_iter) is not int or type(self.tol) not in (int, float) or self.max_iter != 10000 or self.tol != 1e-4:
            raise ValueError("Keep max_iter=10000 and tol=1e-4 fixed for this experiment.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if type(self.allow_unverified) is not bool:
            raise ValueError("allow_unverified must be a boolean.")
        if self.effective_zero_threshold != 1e-8:
            raise ValueError("Keep effective_zero_threshold=1e-8 fixed.")
        if self.output_dir == self.dataset_dir or self.dataset_dir in self.output_dir.parents:
            raise ValueError("Write experiment output outside the dataset directory.")

    @classmethod
    def from_json(cls, path, project_root):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object.")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for name in ("dataset_dir", "reference_metadata", "reference_report", "output_dir"):
            if name not in values or not isinstance(values[name], str) or not values[name].strip():
                raise ValueError(f"Missing or invalid path: {name}")
            value = Path(values[name]).expanduser()
            values[name] = value if value.is_absolute() else Path(project_root) / value
        return cls(**values)

    def to_dict(self):
        values = asdict(self)
        for name in ("dataset_dir", "reference_metadata", "reference_report", "output_dir"):
            values[name] = str(values[name])
        values["c_values"] = list(self.c_values)
        return values
