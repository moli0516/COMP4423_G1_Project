"""CLI: read-only error analysis for B evaluation results.

Usage:
    python scripts/analyze_errors.py \
        --predictions runs/e0_val/predictions.json \
        --manifest runs/val_manifest.json \
        --output-dir runs/error_analysis_val \
        --iou-threshold 0.5 \
        [--cases-per-group 2]

Writes pair_results.csv, event_errors.csv, selected_cases.json and summary.json
into a new output directory. Reports only: it never changes thresholds, filters,
rules or configuration and must not be used to tune anything. JSON is read;
images are never opened, decoded or stat-ed by this tool.

--iou-threshold is required: the predictions file does not record the threshold
used to produce it, so it must be stated on every run. Output is byte-identical
for the same input under the same Python version; float representation is not
guaranteed identical across Python versions.

Exit codes:
    0  analysis written
    1  runtime error
    2  invalid input / refused overwrite
"""

from __future__ import annotations

import argparse
import sys
from math import isfinite
from pathlib import Path

from evaluate_real_pairs import load_manifest

from deskcheck.evaluation.error_analysis import DEFAULT_CASES_PER_GROUP, analyze_errors

STABILITY_NOTE = (
    "Output is byte-identical for the same input under the same Python version; "
    "float representation is not guaranteed identical across Python versions. "
    "The IoU threshold is supplied here because the predictions file does not "
    "record it."
)


def _iou_threshold(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {text!r}") from exc
    if not isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError(
            f"IoU threshold must be finite and in (0, 1], got {text!r}"
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only error analysis for B results.",
        epilog=STABILITY_NOTE,
    )
    parser.add_argument("--predictions", type=Path, required=True,
                        help="predictions.json written by the evaluation runner")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="B-format evaluation manifest (ground truth)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for the analysis outputs")
    parser.add_argument("--iou-threshold", type=_iou_threshold, required=True,
                        help="Event localization IoU threshold, must be in (0, 1]")
    parser.add_argument("--cases-per-group", type=int, default=DEFAULT_CASES_PER_GROUP,
                        help=f"Max selected cases per (status, error_type) group "
                             f"(default {DEFAULT_CASES_PER_GROUP})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        result = analyze_errors(
            args.predictions,
            args.manifest,
            args.output_dir,
            iou_threshold=args.iou_threshold,
            manifest_loader=load_manifest,
            cases_per_group=args.cases_per_group,
        )
    except FileExistsError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    print(f"split={result['split']} release={result['dataset_release']} "
          f"pairs={result['pair_count']} errors={result['error_count']} "
          f"output={result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
