"""Map a COCO-pretrained baseline's class indices onto the six project classes.

The mapping is derived only from the `names` dict stored in the weight file and
matches by *exact class name*. Index positions are never treated as meaning:
they come purely from the checkpoint and may be permuted freely.

Rules:

- Every project class must appear exactly once in the baseline `names` values.
- A missing class is an error, even when a near-variant spelling exists; the
  error names the lookalike so the mismatch is visible instead of silently
  aliased.
- A class name appearing at two indices is a duplicate error.
- A name that has one exact match but also near-variant spellings is ambiguous
  and is rejected.
- Indices must be contiguous from 0, because the reused detector adapter maps
  backend classes by position within its configured name tuple; a gap would
  silently shift that mapping.

Only the public detector contract is used: `BaselineClassDetector` wraps any
`ObjectDetector` and rewrites its class ids, so the existing adapter is not
modified and no inference logic is duplicated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deskcheck.detection.object_detector import ObjectDetector
from deskcheck.detection.yolo_detector import APPROVED_CLASS_NAMES
from deskcheck.domain.detection import Detection
from deskcheck.domain.image_frame import ImageFrame

PROJECT_CLASS_NAMES: tuple[str, ...] = APPROVED_CLASS_NAMES


@dataclass(frozen=True)
class BaselineClassMapping:
    """Baseline class index -> project class id, plus everything left out."""

    class_map: Mapping[int, int]
    project_names: tuple[str, ...]
    baseline_names: Mapping[int, str]
    dropped: tuple[tuple[int, str], ...]

    def remap(self, baseline_index: int) -> int | None:
        """Return the project class id, or None when the class is dropped."""
        return self.class_map.get(baseline_index)

    def table_rows(self) -> list[dict[str, Any]]:
        """One row per baseline class, mapped first, for a human-readable table."""
        rows = [
            {
                "status": "mapped",
                "project_id": project_id,
                "project_name": self.project_names[project_id],
                "baseline_index": baseline_index,
                "baseline_name": self.baseline_names[baseline_index],
            }
            for baseline_index, project_id in sorted(self.class_map.items())
        ]
        rows.extend(
            {
                "status": "dropped",
                "project_id": None,
                "project_name": None,
                "baseline_index": baseline_index,
                "baseline_name": name,
            }
            for baseline_index, name in self.dropped
        )
        return rows


def map_baseline_names(names: Mapping[int, str]) -> BaselineClassMapping:
    """Derive the index mapping from a checkpoint's `names` dict.

    Raises ValueError naming the offending class when a project class is
    missing, duplicated, ambiguous or only present under another spelling.
    """
    _require_valid_names(names)

    by_normalized: dict[str, list[int]] = {}
    for index, name in names.items():
        by_normalized.setdefault(_normalize(name), []).append(index)

    class_map: dict[int, int] = {}
    for project_id, project_name in enumerate(PROJECT_CLASS_NAMES):
        exact = sorted(index for index, name in names.items() if name == project_name)
        if not exact:
            candidates = _similar_names(project_name, names)
            hint = (
                f"; found similar spellings {candidates} but aliases are not allowed"
                if candidates else ""
            )
            raise ValueError(
                f"baseline names are missing project class {project_name!r}{hint}"
            )
        if len(exact) > 1:
            raise ValueError(
                f"baseline class {project_name!r} is duplicated at indices {exact}"
            )
        index = exact[0]
        variants = sorted(
            names[index] for index in by_normalized.get(_normalize(project_name), [])
            if names[index] != project_name
        )
        if variants:
            raise ValueError(
                f"baseline class {project_name!r} is ambiguous: exact match at index "
                f"{index} but near-variant spellings {variants} also exist; "
                "aliases are not allowed"
            )
        class_map[index] = project_id

    if sorted(class_map.values()) != list(range(len(PROJECT_CLASS_NAMES))):
        raise ValueError("baseline mapping is not one-to-one onto the project classes")

    dropped = tuple(
        sorted((index, name) for index, name in names.items() if index not in class_map)
    )
    return BaselineClassMapping(
        class_map=dict(sorted(class_map.items())),
        project_names=PROJECT_CLASS_NAMES,
        baseline_names=dict(sorted(names.items())),
        dropped=dropped,
    )


class BaselineClassDetector(ObjectDetector):
    """Apply `mapping` to another detector's class ids, dropping other classes."""

    def __init__(self, *, detector: ObjectDetector, mapping: BaselineClassMapping) -> None:
        if not callable(getattr(detector, "predict", None)):
            raise TypeError("detector must expose predict(frame)")
        if not isinstance(mapping, BaselineClassMapping):
            raise TypeError("mapping must be a BaselineClassMapping")
        self._detector = detector
        self._mapping = mapping

    def predict(self, image: ImageFrame) -> list[Detection]:
        """Return project-class detections in the inner adapter's pixel space."""
        mapped: list[Detection] = []
        for detection in self._detector.predict(image):
            project_id = self._mapping.remap(detection.class_id)
            if project_id is None:
                continue
            mapped.append(
                Detection(
                    detection_id=detection.detection_id,
                    class_id=project_id,
                    confidence=detection.confidence,
                    box=detection.box,
                )
            )
        mapped.sort(key=lambda item: (item.class_id, -item.confidence, item.detection_id))
        return mapped


def _require_valid_names(names: Mapping[int, str]) -> None:
    if not isinstance(names, Mapping):
        raise TypeError("names must be a mapping of integer index to class name")
    if not names:
        raise ValueError("baseline names must not be empty")
    for index, name in names.items():
        if type(index) is not int or index < 0:
            raise ValueError(f"baseline name index must be a nonnegative integer, got {index!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"baseline name at index {index} must be a nonempty string")
    expected = list(range(len(names)))
    if sorted(names) != expected:
        raise ValueError(
            "baseline name indices must be contiguous from 0 so the reused detector "
            f"adapter cannot shift them; got {sorted(names)[:8]}"
        )


def _normalize(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def _similar_names(project_name: str, names: Mapping[int, str]) -> list[str]:
    """Report lookalike spellings so a missing class is never silently aliased."""
    target = _normalize(project_name)
    similar = set()
    for name in names.values():
        candidate = _normalize(name)
        if candidate == target or (len(target) >= 3 and (
            candidate.startswith(target) or target.startswith(candidate)
        )):
            similar.add(name)
    return sorted(similar)
