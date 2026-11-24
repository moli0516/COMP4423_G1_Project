"""Owner A: CLI wrapper for dataset validation.

Usage:
    python scripts/validate_dataset.py \
        --annotations data/annotations/pairs.jsonl \
        --manifest data/manifests/splits.json

Exit codes (per 05_required_api_list.pdf):
    0  audit passed
    1  runtime error
    2  invalid input / config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a DeskCheck annotation + split manifest.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to data/annotations/pairs.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to data/manifests/splits.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON to stdout.",
    )
    return parser


def _report_to_dict(report) -> dict:
    return {
        "ok": report.ok,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "pair_counts": report.pair_counts,
        "unique_image_counts": report.unique_image_counts,
        "group_counts": report.group_counts,
        "class_coverage": report.class_coverage,
        "status_coverage": report.status_coverage,
        "independent_no_change_counts": report.independent_no_change_counts,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from deskcheck.data.dataset_catalog import load_catalog
        from deskcheck.data.dataset_validator import audit
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        pairs, manifest = load_catalog(args.annotations, args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    try:
        report = audit(pairs, manifest)
    except Exception as exc:  # noqa: BLE001 - surface any runtime failure
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_report_to_dict(report), indent=2))
    else:
        print(f"dataset: {args.annotations}")
        print(f"manifest: {args.manifest}")
        print(f"pairs: {report.pair_counts}")
        print(f"unique images: {report.unique_image_counts}")
        print(f"capture groups: {report.group_counts}")
        print(f"class coverage: {report.class_coverage}")
        print(f"status coverage: {report.status_coverage}")
        print(f"independent no-change: {report.independent_no_change_counts}")
        if report.warnings:
            print("warnings:")
            for w in report.warnings:
                print(f"  - {w}")
        if report.errors:
            print("ERRORS:")
            for e in report.errors:
                print(f"  - {e}")
        else:
            print("OK")

    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
