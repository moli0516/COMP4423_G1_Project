"""Strict scalar parsing for the shared YOLO dataset layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetLayout:
    path: str
    train: str
    val: str


def parse_supported_dataset_layout(text: str) -> DatasetLayout:
    values: dict[str, str | None] = {}
    present: set[str] = set()
    for raw in text.splitlines():
        for key in ("path", "train", "val", "train_labels", "val_labels"):
            prefix = key + ":"
            if not raw.startswith(prefix):
                continue
            if key in present:
                raise ValueError(f"Duplicate YAML key {key!r} is unsupported")
            present.add(key)
            value = raw[len(prefix):].strip().strip("\"'")
            if value.startswith(("[", "-")):
                raise ValueError(
                    f"Unsupported YAML list value for {key!r}; use the shared directory layout"
                )
            values[key] = value or None

    path = values.get("path")
    if path is None:
        raise ValueError("dataset.yaml path is required by the shared trainer")
    train = values.get("train")
    val = values.get("val")
    for split, target in (("train", train), ("val", val)):
        if target is None:
            raise ValueError(f"dataset.yaml requires an explicit {split}: target")
        expected = f"images/{split}"
        if target != expected:
            raise ValueError(f"dataset {split} target must be {expected}")
    if {"train_labels", "val_labels"} & present:
        raise ValueError("Custom label targets are unsupported by the shared trainer")

    return DatasetLayout(path=path, train=train, val=val)
