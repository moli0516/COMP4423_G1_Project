"""CLI: detector mAP evaluation on an isolated copy of the dataset.

Usage:
    python scripts/evaluate_detector_map.py \
        --weights path/to/best.pt \
        --dataset-yaml path/to/dataset.yaml \
        --split val \
        --output artifacts/detector_map_val.json \
        --copy-dir artifacts/dataset_copy_val

The dataset is copied first and the backend only ever validates the copy.
split=test is the held-out final evaluation and requires --confirm-final-test.

Exit codes:
    0  report written
    1  runtime error (backend failure)
    2  invalid input / refused overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deskcheck.evaluation.detector_map import evaluate_detector_map

TEST_WARNING = (
    "WARNING: split=test is the held-out final evaluation. Run it once, after the "
    "configuration is frozen; re-running it after tuning invalidates the result."
)


def _real_validator(*, weights: Path, dataset_yaml: Path, split: str) -> dict:
    """Run ultralytics detection validation. Real backend wiring lives only here."""
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install -e '.[vision]'"
        ) from exc

    metrics = YOLO(str(weights)).val(data=str(dataset_yaml), split=split)
    names = metrics.names
    class_names = [str(names[index]) for index in sorted(names)]
    class_index = [int(index) for index in metrics.box.ap_class_index]
    ap50 = [float(value) for value in metrics.box.ap50]
    ap50_95 = [float(value) for value in metrics.box.ap]
    per_class = {
        class_names[index]: {"ap50": value50, "ap50_95": value95}
        for index, value50, value95 in zip(class_index, ap50, ap50_95, strict=True)
    }
    results = metrics.results_dict
    return {
        "backend_name": "ultralytics",
        "backend_version": str(ultralytics.__version__),
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
        "class_names": class_names,
        "per_class": per_class,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate detector mAP on a copy of the YOLO dataset.",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Detector checkpoint")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--split", choices=("val", "test"), required=True,
                        help="Dataset split to validate")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--copy-dir", type=Path, required=True,
                        help="New directory that receives the dataset copy")
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="Required to evaluate the held-out test split")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.split == "test":
        print(TEST_WARNING, file=sys.stderr)

    try:
        report = evaluate_detector_map(
            weights=args.weights,
            dataset_yaml=args.dataset_yaml,
            split=args.split,
            output=args.output,
            copy_dir=args.copy_dir,
            validator=_real_validator,
            confirm_final_test=args.confirm_final_test,
        )
    except FileExistsError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except (OSError, TypeError, RuntimeError) as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    print(f"split={report['split']} map50={report['map50']:.6f} "
          f"map50_95={report['map50_95']:.6f}")
    print(f"backend={report['backend_name']} {report['backend_version']} "
          f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
