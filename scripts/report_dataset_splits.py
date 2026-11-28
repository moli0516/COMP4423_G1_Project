"""CLI: write a read-only dataset split report.

Usage:
    python scripts/report_dataset_splits.py \
        --pairs /path/to/pairs.jsonl \
        --splits /path/to/splits.json \
        --dataset-release v1 \
        --output-dir artifacts/dataset_report

Writes `report.json` and `report.md` into --output-dir, which must not exist
yet. Nothing is written when the run fails. Annotation metadata only: no image
is opened, decoded, stat-ed or hashed, so this can run on every split including
test.

`--expect-pairs 14,4,6` (train,val,test) and `--expect-objects 269` add a
separate "differences from expected" section to report.md. They are
informational: they never change a check, the JSON report or the exit code.

Exit codes:
    0  report written (read the printed status; a failing check still produces
       a report, which is the deliverable)
    1  runtime error
    2  invalid input, release mismatch, a structural violation rejected by the
       strict loader, or an existing output directory
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from deskcheck.evaluation.dataset_report import (
    build_dataset_report,
    render_markdown,
    report_to_json,
)


def _pair_counts(text: str) -> tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected three comma-separated counts (train,val,test), got {text!r}"
        )
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"counts must be integers, got {text!r}") from exc
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"counts must be non-negative, got {text!r}")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a read-only dataset split report from annotation metadata.",
    )
    parser.add_argument("--pairs", type=Path, required=True,
                        help="Released pair annotations (pairs.jsonl), read only")
    parser.add_argument("--splits", type=Path, required=True,
                        help="Released split manifest (splits.json), read only")
    parser.add_argument("--dataset-release", required=True,
                        help="Dataset release name; must match splits.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory that will receive report.json and report.md")
    parser.add_argument("--expect-pairs", type=_pair_counts, default=None,
                        help="Optional expected train,val,test pair counts, e.g. 14,4,6")
    parser.add_argument("--expect-objects", type=int, default=None,
                        help="Optional expected object count in the pairs file")
    return parser


def _write_outputs(output_dir: Path, files: dict[str, str]) -> None:
    """Create output_dir with `files` in one rename; leave nothing on failure."""
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for name, text in files.items():
            (staging / name).write_bytes(text.encode("utf-8"))
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite existing output directory: {target}"
            )
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        report = build_dataset_report(args.pairs, args.splits, args.dataset_release)
        markdown = render_markdown(
            report,
            expect_pairs=args.expect_pairs,
            expect_objects=args.expect_objects,
        )
        _write_outputs(
            args.output_dir,
            {"report.json": report_to_json(report), "report.md": markdown},
        )
    except FileExistsError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    summary = report["summary"]
    totals = report["splits"]["total"]
    per_split = ",".join(
        f"{name}={report['splits'][name]['pair_count']}"
        for name in ("train", "val", "test")
    )
    print(
        f"status={'pass' if summary['ok'] else 'issues'} "
        f"checks={summary['check_count']} passed={summary['passed']} "
        f"failed={summary['failed']} unverified={summary['unverified']} "
        f"pairs={totals['pair_count']} ({per_split}) objects={totals['object_count']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
