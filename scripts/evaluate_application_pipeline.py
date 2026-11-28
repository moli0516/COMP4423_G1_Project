"""Score saved post-B-filter detections as an application-pipeline diagnostic.

This CLI never imports or runs the ML backend. It reads a completed evaluation
run's saved post-B-filter detections, an explicit evaluation config, and a
per-image ground-truth file, then writes a new report directory.

The result is an application-pipeline precision/recall/F1 diagnostic, not a
detector metric: B's DetectionFilter keeps at most one detection per class per
image, so recall is non-increasing relative to raw detector output. A true
detector metric needs pre-filter detections, which are not saved yet.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.evaluation.application_pipeline_evaluator import (
    ApplicationPipelineConfig,
    evaluate_saved_pipeline_run,
)
from deskcheck.evaluation.failure_record import (
    build_failure_record,
    build_rerun_link,
    new_run_id,
    write_failure_record,
)


def _publish(report: dict, target: Path) -> Path:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite pipeline report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / "application_pipeline_metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite pipeline report: {target}")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "application_pipeline_metrics.json"


def _failure_record(exc: Exception, *, run_id: str, config, args, rerun_of: list[dict]):
    config_sha = config.config_sha256 if config is not None else None
    return build_failure_record(
        run_id=run_id,
        reason=str(exc) or type(exc).__name__,
        exception_type=type(exc).__name__,
        context={"stage": "application_pipeline_evaluation", "pair_id": None, "image_id": None},
        input_identities={
            "manifest_sha256": None,
            "checkpoint_sha256": None,
            "config_sha256": config_sha,
            "dataset_release": None,
            "split": None,
        },
        source_identity={
            "source_revision": None,
            "source_dirty": None,
            "source_dirty_paths": None,
        },
        evaluation_config={
            "evaluation_run": str(args.run),
            "ground_truth": str(args.ground_truth),
            "output": str(args.output),
            "confidence_threshold": getattr(config, "confidence_threshold", None),
            "nms_iou_threshold": getattr(config, "nms_iou_threshold", None),
        },
        command=list(sys.argv),
        rerun_of=rerun_of,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Completed evaluation run containing post-B-filter image_detections.json",
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failure-output", type=Path, default=None)
    parser.add_argument("--rerun-of", type=Path, action="append", default=[])
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)

    target = args.output.absolute()
    if target.exists() or target.is_symlink():
        print(f"Refusing to overwrite pipeline report: {target}", file=sys.stderr)
        return 2
    if args.rerun_of and not (args.rerun_reason and args.rerun_reason.strip()):
        print("--rerun-reason is required with --rerun-of", file=sys.stderr)
        return 2

    try:
        rerun_of = [build_rerun_link(path, args.rerun_reason) for path in args.rerun_of]
    except (OSError, ValueError, TypeError) as error:
        print(f"Rerun link is invalid: {error}", file=sys.stderr)
        return 2

    run_id = new_run_id()
    config = None
    try:
        config = ApplicationPipelineConfig.load(args.config)
        report = evaluate_saved_pipeline_run(config, args.run, args.ground_truth)
        report["run_id"] = run_id
        report["rerun_of"] = rerun_of
        written = _publish(report, target)
        print(f"Application-pipeline report written: {written}")
        return 0
    except Exception as error:  # noqa: BLE001 - every failure must be recorded
        failure_output = args.failure_output
        if failure_output is None:
            failure_output = target.with_name(f"{target.name}.failure-{run_id}")
        try:
            record = _failure_record(
                error, run_id=run_id, config=config, args=args, rerun_of=rerun_of
            )
            write_failure_record(record, failure_output)
        except Exception as record_error:  # noqa: BLE001 - report the original error too
            print(
                f"Application-pipeline evaluation failed: {error}; "
                f"failure record could not be written: {record_error}",
                file=sys.stderr,
            )
            return 1
        print(f"Application-pipeline evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
