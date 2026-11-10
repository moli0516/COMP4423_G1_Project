"""Owner A: CLI wrapper for YOLO dataset export.

Usage:
    python scripts/export_yolo_dataset.py \
        --annotations data/annotations/pairs.jsonl \
        --manifest data/manifests/splits.json \
        --output data/exports/yolo

Exit codes:
    0  export written
    1  runtime error
    2  invalid input / config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export DeskCheck annotations to YOLO label format.",
    )
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Path to data/annotations/pairs.jsonl")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to data/manifests/splits.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for images/labels/dataset.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from deskcheck.data.dataset_catalog import load_catalog
        from deskcheck.data.yolo_exporter import export
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        pairs, manifest = load_catalog(args.annotations, args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    try:
        summary = export(pairs, manifest, args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {summary.output_dir}")
    print(f"pairs processed: {summary.pairs_processed}")
    print(f"images written:  {summary.images_written}")
    print(f"labels written:  {summary.labels_written}")
    print(f"empty labels:    {summary.empty_labels_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
