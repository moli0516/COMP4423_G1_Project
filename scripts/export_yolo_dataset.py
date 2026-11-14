"""Owner A: CLI wrapper for YOLO dataset export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export DeskCheck annotations to YOLO label + image layout.",
    )
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Path to data/annotations/pairs.jsonl")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to data/manifests/splits.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for images/labels/dataset.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Allow writing into a non-empty output directory.")
    parser.add_argument("--movement-threshold", type=float, default=None,
                        help="Optional frozen tau; enables moved/unchanged "
                             "consistency check in the audit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from deskcheck.data.dataset_catalog import load_catalog
        from deskcheck.data.dataset_validator import audit
        from deskcheck.data.yolo_exporter import export
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        pairs, manifest = load_catalog(args.annotations, args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    # Audit BEFORE touching the output directory so a failed audit
    # doesn't leave a partially written export behind.
    report = audit(pairs, manifest, movement_threshold=args.movement_threshold)
    if not report.ok:
        print("audit failed; refusing to export", file=sys.stderr)
        for e in report.errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    try:
        summary = export(pairs, manifest, args.output, overwrite=args.force)
    except FileExistsError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {summary.output_dir}")
    print(f"pairs processed: {summary.pairs_processed}")
    print(f"images written:  {summary.images_written}")
    print(f"labels written:  {summary.labels_written}")
    print(f"empty labels:    {summary.empty_labels_written}")
    print(f"shared dedup:    {summary.shared_images_deduplicated}")
    if report.warnings:
        for w in report.warnings:
            print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
