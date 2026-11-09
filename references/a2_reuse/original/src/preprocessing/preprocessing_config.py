from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

RESIZE_MODES = ("squash", "center_crop", "pad")


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 256
    resize_mode: str = "squash"
    pad_value: int = 0

    def __post_init__(self):
        if type(self.image_size) is not int or self.image_size < 16:
            raise ValueError("image_size must be an integer of at least 16.")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}.")
        if type(self.pad_value) is not int or not 0 <= self.pad_value <= 255:
            raise ValueError("pad_value must be an integer between 0 and 255.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, Mapping):
            raise ValueError("Preprocessing settings must be a mapping.")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown preprocessing settings: {sorted(map(str, unknown))}")
        return cls(**dict(values))


DEFAULT_CONFIG = PreprocessingConfig()
