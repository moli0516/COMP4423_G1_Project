"""Owner A: CLI wrapper for whole-group dataset splitting.

Usage:
    python scripts/split_dataset.py \
        --annotations data/annotations/pairs.jsonl \
        --output data/manifests/splits.json

Exit codes:
    0  split written (audit passed)
    1  runtime error
    2  invalid input / config / infeasible coverage / failed audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a DeskCheck dataset into train/val/test by capture group.",
    )
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Path to data/annotations/pairs.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the split manifest JSON.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic split seed (default: 42).")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--movement-threshold", type=float, default=None,
                        help="Optional frozen tau; enables moved/unchanged "
                             "consistency check in the audit.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output manifest. Without "
                             "this flag, existing output is rejected.")
    return parser


def _manifest_to_payload(manifest, report) -> dict:
    return {
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "dataset_release": manifest.dataset_release,
        "splits": {name: list(ids) for name, ids in manifest.splits.items()},
        "coverage": {
            "classes_per_split": report.class_coverage,
            "statuses_per_split": report.status_coverage,
            "independent_no_change_pairs": report.independent_no_change_counts,
        },
    }


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from deskcheck.data.dataset_catalog import load_annotations
        from deskcheck.data.dataset_splitter import split
        from deskcheck.data.dataset_validator import audit
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Refuse to overwrite by default.
    if args.output.exists() and not args.force:
        print(f"refusing to overwrite existing manifest: {args.output} "
              f"(use --force to replace)", file=sys.stderr)
        return 2

    try:
        pairs = load_annotations(args.annotations)
    except (FileNotFoundError, ValueError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if any(r < 0 for r in ratios.values()):
        print("invalid input: ratios must be non-negative", file=sys.stderr)
        return 2

    try:
        manifest = split(list(pairs.values()), seed=args.seed,
                         target_ratios=ratios)
    except ValueError as exc:
        print(f"split failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    report = audit(pairs, manifest, movement_threshold=args.movement_threshold)
    if not report.ok:
        print("audit failed; refusing to publish split manifest", file=sys.stderr)
        for e in report.errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    payload = _manifest_to_payload(manifest, report)
    try:
        _atomic_write(args.output, json.dumps(payload, indent=2) + "\n")
    except OSError as exc:
        print(f"write failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {args.output}")
    print(f"pairs: {report.pair_counts}")
    print(f"unique images: {report.unique_image_counts}")
    print(f"capture groups: {report.group_counts}")
    if report.warnings:
        for w in report.warnings:
            print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
