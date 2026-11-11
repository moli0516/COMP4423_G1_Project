"""Owner A: export annotations to YOLO layout.

Writes both label files AND copies image bytes into images/{split}/, so the
exported tree is directly usable by Ultralytics YOLO and by any trainer that
expects `dataset.yaml` to point at real files.

Design decisions (per B's review, 11 Nov 2026):
- Export key: `{pair_id}__{side}` where side is 'before' or 'after'.
  Unique across the whole dataset because pair_id is unique per pair.
- Image bytes are copied (not symlinked, not assumed external).
- Shared image_id across pairs: allowed only if path+dimensions agree.
  Otherwise the export aborts before writing anything.
- Non-empty output directory: aborted unless `overwrite=True` is passed.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from deskcheck.data.dataset_catalog import AnnotatedPair, SplitManifest

DEFAULT_CLASS_NAMES = ("bottle", "cup", "keyboard", "mouse", "book", "scissors")


@dataclass(frozen=True)
class ExportSummary:
    output_dir: Path
    images_written: int
    labels_written: int
    pairs_processed: int
    empty_labels_written: int
    shared_images_deduplicated: int


def _export_key(pair_id: str, side: str) -> str:
    """Filesystem-safe unique key for a pair-side image."""
    assert side in ("before", "after"), side
    return f"{pair_id}__{side}"


def _yolo_line(class_id: int, bbox_xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _check_shared_image_consistency(
    pairs: dict[str, AnnotatedPair],
) -> dict[str, tuple[str, int, int, tuple[tuple[int, tuple[float, float, float, float]], ...]]]:
    """Group every image_id across pairs; verify agreement on path/dims/annotations.

    Returns {image_id: (source_path, width, height, annotation_fingerprint)}.
    Raises ValueError on disagreement.
    """
    seen: dict[str, tuple[str, int, int, tuple]] = {}
    for pair in pairs.values():
        for side, image_id, source_path in (
            ("before", pair.before_image_id, pair.before_image_path),
            ("after", pair.after_image_id, pair.after_image_path),
        ):
            objects = pair.manual_before_objects if side == "before" else pair.manual_after_objects
            fingerprint = tuple(
                (obj["class_id"], tuple(float(v) for v in obj["bbox_xyxy"]))
                for obj in objects
            )
            record = (source_path, pair.width, pair.height, fingerprint)
            if image_id in seen:
                existing = seen[image_id]
                if (existing[0] != record[0]
                        or existing[1] != record[1]
                        or existing[2] != record[2]
                        or existing[3] != record[3]):
                    raise ValueError(
                        f"shared image_id {image_id!r} has conflicting representations:\n"
                        f"  first:  path={existing[0]!r} dims=({existing[1]},{existing[2]})\n"
                        f"  second: path={record[0]!r} dims=({record[1]},{record[2]})\n"
                        f"Reject the export; resolve the inconsistency in annotations first."
                    )
            else:
                seen[image_id] = record
    return seen


def _ensure_empty_or_overwrite(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    has_content = any(output_dir.iterdir())
    if has_content and not overwrite:
        raise FileExistsError(
            f"export directory already has content: {output_dir}. "
            f"Pass overwrite=True (CLI: --force) to replace it."
        )


def _copy_image(source_path: Path, dest_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(
            f"image file missing: {source_path}. "
            f"The exporter requires actual image bytes to be present."
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)


def export(
    pairs: dict[str, AnnotatedPair],
    manifest: SplitManifest,
    output_dir: Path,
    *,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    overwrite: bool = False,
) -> ExportSummary:
    """Write YOLO images+labels+dataset.yaml. Does not mutate `pairs`."""
    output_dir = Path(output_dir)
    _ensure_empty_or_overwrite(output_dir, overwrite)

    # Verify shared-image consistency BEFORE writing anything.
    # Side effect only: raises ValueError on conflicting representations.
    _check_shared_image_consistency(pairs)

    # Build plan: for each split, list of (image_id, export_key, side, pair).
    plan: dict[str, list[tuple[str, str, str, AnnotatedPair]]] = {
        name: [] for name in manifest.splits
    }
    seen_in_split: dict[str, set[str]] = {name: set() for name in manifest.splits}
    shared_dedup_count = 0

    for split_name, pair_ids in manifest.splits.items():
        for pair_id in pair_ids:
            pair = pairs[pair_id]
            for side, image_id in (
                ("before", pair.before_image_id),
                ("after", pair.after_image_id),
            ):
                if image_id in seen_in_split[split_name]:
                    shared_dedup_count += 1
                    continue
                seen_in_split[split_name].add(image_id)
                key = _export_key(pair_id, side)
                plan[split_name].append((image_id, key, side, pair))

    # Write dataset.yaml first (so failures later don't leave a partial tree).
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names: {list(class_names)}\n"
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    images_written = 0
    labels_written = 0
    empty_labels = 0

    for split_name, entries in plan.items():
        images_dir = output_dir / "images" / split_name
        labels_dir = output_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_id, key, side, pair in entries:
            source_path = Path(
                pair.before_image_path if side == "before" else pair.after_image_path
            )
            extension = source_path.suffix or ".jpg"
            dest_image = images_dir / f"{key}{extension}"
            _copy_image(source_path, dest_image)
            images_written += 1

            objects = (
                pair.manual_before_objects if side == "before"
                else pair.manual_after_objects
            )
            lines = [
                _yolo_line(obj["class_id"], obj["bbox_xyxy"], pair.width, pair.height)
                for obj in objects
            ]
            text = "\n".join(lines) + ("\n" if lines else "")
            (labels_dir / f"{key}.txt").write_text(text, encoding="utf-8")
            labels_written += 1
            if not lines:
                empty_labels += 1

    return ExportSummary(
        output_dir=output_dir,
        images_written=images_written,
        labels_written=labels_written,
        pairs_processed=sum(len(ids) for ids in manifest.splits.values()),
        empty_labels_written=empty_labels,
        shared_images_deduplicated=shared_dedup_count,
    )
