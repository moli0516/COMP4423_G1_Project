"""Owner A: export annotations to YOLO label format.

One label file per unique image. Shared images across pairs are written once.
Preserves split isolation. Does not copy or open image bytes.
"""

from __future__ import annotations

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


def _yolo_line(class_id: int, bbox_xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _write_split(
    split_name: str,
    pair_ids: tuple[str, ...],
    pairs: dict[str, AnnotatedPair],
    output_dir: Path,
) -> tuple[int, int, int]:
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    seen_images: dict[str, tuple[AnnotatedPair, str]] = {}
    for pair_id in pair_ids:
        pair = pairs[pair_id]
        for image_id, side in (
            (pair.before_image_id, "before"),
            (pair.after_image_id, "after"),
        ):
            if image_id in seen_images:
                continue
            seen_images[image_id] = (pair, side)

    labels_written = 0
    empty_labels = 0
    for image_id, (pair, side) in seen_images.items():
        if side == "before":
            objects = pair.manual_before_objects
            source_path = pair.before_image_path
        else:
            objects = pair.manual_after_objects
            source_path = pair.after_image_path
        stem = Path(source_path).stem
        lines = [
            _yolo_line(obj["class_id"], obj["bbox_xyxy"], pair.width, pair.height)
            for obj in objects
        ]
        text = "\n".join(lines) + ("\n" if lines else "")
        (labels_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        labels_written += 1
        if not lines:
            empty_labels += 1
    return len(seen_images), labels_written, empty_labels


def export(
    pairs: dict[str, AnnotatedPair],
    manifest: SplitManifest,
    output_dir: Path,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> ExportSummary:
    """Write YOLO labels tree plus dataset.yaml. Does not copy image bytes."""
    output_dir = Path(output_dir)
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

    total_images = 0
    total_labels = 0
    total_empty = 0
    total_pairs = 0
    for split_name, pair_ids in manifest.splits.items():
        imgs, labels, empty = _write_split(split_name, pair_ids, pairs, output_dir)
        total_images += imgs
        total_labels += labels
        total_empty += empty
        total_pairs += len(pair_ids)
    return ExportSummary(
        output_dir=output_dir,
        images_written=total_images,
        labels_written=total_labels,
        pairs_processed=total_pairs,
        empty_labels_written=total_empty,
    )
