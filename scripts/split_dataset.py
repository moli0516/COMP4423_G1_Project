"""Owner A: CLI wrapper for whole-group dataset splitting.

Usage:
    python scripts/split_dataset.py \
        --annotations data/annotations/pairs.jsonl \
        --seed 42 \
        --output data/manifests/splits.json

Exit codes:
    0  split written
    1  runtime error
    2  invalid input / config (including infeasible coverage)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a DeskCheck dataset into train/val/test by capture group.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to data/annotations/pairs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the split manifest JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed (default: 42).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
        help="Train ratio (default: 0.60).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio (default: 0.20).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Test ratio (default: 0.20).",
    )
    return parser


def _write_manifest(path: Path, manifest, report) -> None:
    payload = {
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from deskcheck.data.dataset_catalog import load_annotations
        from deskcheck.data.dataset_splitter import split
        from deskcheck.data.dataset_validator import audit
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
        manifest = split(list(pairs.values()), seed=args.seed, target_ratios=ratios)
    except ValueError as exc:
        print(f"split failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    report = audit(pairs, manifest)
    _write_manifest(args.output, manifest, report)

    print(f"wrote {args.output}")
    print(f"pairs: {report.pair_counts}")
    print(f"unique images: {report.unique_image_counts}")
    print(f"capture groups: {report.group_counts}")
    if report.errors:
        print("WARNING: audit reported errors on this split:")
        for e in report.errors:
            print(f"  - {e}")
        print("(manifest written anyway; re-run validate_dataset.py to gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
