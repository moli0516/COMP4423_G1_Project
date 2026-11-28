"""CLI: compare a COCO-pretrained baseline against E0 on the val split.

Both models run through the same existing pipeline and settings: the shared
detector adapter, `DetectionFilter`, `ObjectMatcher`, `ChangeAnalyzer`,
`EventEvaluator`, `NoChangeEvaluator` and `EvaluationRunner`. Only the baseline
needs an extra step, because its class indices differ from the project classes:
`BaselineClassDetector` applies the mapping derived from the checkpoint's own
`names` dict before filtering and matching.

VAL ONLY, AND NOT HELD-OUT: E0's checkpoint and decision threshold were chosen
on this same val split, so the comparison is biased toward E0. Never describe
these numbers as held-out or as an unbiased model comparison.

Both weight files are verified against the expected SHA-256 before anything is
loaded, the output directory must not exist, and a failed run leaves nothing
behind. Nothing here changes thresholds, filters, rules or configuration.

Usage:
    python scripts/compare_baseline.py \
        --baseline-weights /path/to/yolov8n.pt \
        --baseline-sha256 <64 hex> \
        --e0-weights artifacts/best.pt \
        --e0-sha256 <64 hex> \
        --config configs/app.json \
        --manifest runs/val_manifest.json \
        --output-dir runs/baseline_vs_e0 \
        --event-iou-threshold 0.5

Exit codes:
    0  comparison written
    1  runtime error
    2  invalid input / refused overwrite
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from evaluate_real_pairs import load_manifest

from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.real_settings import RealSettings, sha256_file
from deskcheck.detection.yolo_detector import YoloConfig, YoloDetector
from deskcheck.evaluation.baseline_mapping import (
    BaselineClassDetector,
    BaselineClassMapping,
    map_baseline_names,
)
from deskcheck.evaluation.evaluation_runner import EvaluationRunner

VAL_SPLIT = "val"
STATUSES = ("added", "removed", "moved", "unchanged")
MAPPING_CSV_FIELDS = (
    "status", "project_id", "project_name", "baseline_index", "baseline_name",
)
SELECTION_BIAS = (
    "E0's checkpoint and decision threshold were selected on this val split, so "
    "this comparison is biased toward E0 and is NOT held-out."
)


@dataclass(frozen=True)
class BackendInfo:
    """What the weight file itself says about its classes and backend."""

    names: Mapping[int, str]
    version: str


def read_backend_info(weights: Path) -> BackendInfo:
    """Read `names` and the ultralytics version from a weight file."""
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install -e '.[vision]'"
        ) from exc

    model = YOLO(str(weights))
    raw_names = getattr(model, "names", None)
    if isinstance(raw_names, Mapping):
        names = dict(raw_names)
    elif isinstance(raw_names, (list, tuple)):
        names = dict(enumerate(raw_names))
    else:
        raise TypeError("weight file must expose an indexed class map")
    return BackendInfo(names=names, version=str(ultralytics.__version__))


def build_comparison_service(
    *,
    weights: Path,
    settings: object,
    info: BackendInfo,
    mapping: BaselineClassMapping | None,
) -> ComparisonService:
    """Build the shared pipeline for E0 (mapping None) or for the baseline."""
    if mapping is None:
        if Path(weights).resolve() != settings.checkpoint_path:
            raise ValueError(
                f"e0 weights {weights} disagree with the configured checkpoint "
                f"{settings.checkpoint_path}"
            )
        return ComparisonService.for_real(settings)

    ordered_names = tuple(info.names[index] for index in sorted(info.names))
    inner = YoloDetector(YoloConfig(
        checkpoint_path=Path(weights),
        class_names=ordered_names,
        confidence_threshold=settings.confidence_threshold,
        nms_iou_threshold=settings.nms_iou_threshold,
        image_size=settings.imgsz,
        device="cpu",
    ))
    return ComparisonService(
        config=settings,
        detector=BaselineClassDetector(detector=inner, mapping=mapping),
    )


def run_comparison(
    *,
    baseline_weights: str | Path,
    baseline_sha256: str,
    e0_weights: str | Path,
    e0_sha256: str,
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    event_iou_threshold: float,
    manifest_loader: Callable[[Path], dict] = load_manifest,
    settings_loader: Callable[[Path], object] = RealSettings.load,
    read_backend: Callable[[Path], BackendInfo] = read_backend_info,
    build_service: Callable[..., ComparisonService] = build_comparison_service,
) -> dict:
    """Run both models on one val manifest and write one output directory.

    Refuses a non-val manifest, hash mismatches, an existing output directory and
    unreadable inputs; publishes only after both runs and every artifact succeed.
    """
    threshold = _require_iou_threshold(event_iou_threshold)
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {target}")

    manifest = manifest_loader(Path(manifest_path))
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest loader must return a mapping")
    split = manifest["split"]
    if split != VAL_SPLIT:
        raise ValueError(
            f"refusing non-val manifest: split is {split!r}, only {VAL_SPLIT!r} is allowed"
        )

    baseline_path = Path(baseline_weights).resolve()
    e0_path = Path(e0_weights).resolve()
    baseline_digest = _require_weights(baseline_path, baseline_sha256, "baseline")
    e0_digest = _require_weights(e0_path, e0_sha256, "e0")

    settings = settings_loader(Path(config_path))
    if Path(settings.checkpoint_path) != e0_path:
        raise ValueError(
            f"e0 weights {e0_path} disagree with the configured checkpoint "
            f"{settings.checkpoint_path}"
        )
    if settings.checkpoint_sha256 != e0_digest:
        raise ValueError(
            "e0 weights SHA-256 disagrees with the hash recorded in the config: "
            f"{e0_digest} vs {settings.checkpoint_sha256}"
        )

    baseline_info = read_backend(baseline_path)
    e0_info = read_backend(e0_path)
    mapping = map_baseline_names(baseline_info.names)
    services = {
        "baseline": build_service(
            weights=baseline_path, settings=settings, info=baseline_info, mapping=mapping
        ),
        "e0": build_service(
            weights=e0_path, settings=settings, info=e0_info, mapping=None
        ),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        metrics: dict[str, dict] = {}
        for role, service in services.items():
            result = EvaluationRunner(service, iou_threshold=threshold).run(
                pairs=list(manifest["pairs"]),
                ground_truth=manifest["ground_truth"],
                eligible_no_change_pair_ids=list(manifest["eligible_no_change_pair_ids"]),
                output_dir=staging / role,
                metadata=_run_metadata(
                    role=role,
                    weights=baseline_path if role == "baseline" else e0_path,
                    digest=baseline_digest if role == "baseline" else e0_digest,
                    settings=settings,
                    info=baseline_info if role == "baseline" else e0_info,
                    manifest=manifest,
                    threshold=threshold,
                ),
            )
            metrics[role] = result["metrics"]

        summary = _comparison_payload(
            mapping=mapping,
            baseline_path=baseline_path,
            baseline_digest=baseline_digest,
            baseline_info=baseline_info,
            e0_path=e0_path,
            e0_digest=e0_digest,
            e0_info=e0_info,
            settings=settings,
            manifest=manifest,
            threshold=threshold,
            metrics=metrics,
        )
        (staging / "baseline_class_mapping.json").write_text(
            _json_text(_mapping_payload(mapping, baseline_path, baseline_digest)),
            encoding="utf-8",
        )
        (staging / "baseline_class_mapping.csv").write_text(
            _mapping_csv(mapping), encoding="utf-8"
        )
        (staging / "comparison.json").write_text(_json_text(summary), encoding="utf-8")
        (staging / "comparison.csv").write_text(_comparison_csv(metrics), encoding="utf-8")

        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {target}")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "split": split,
        "dataset_release": manifest["dataset_release"],
        "pair_count": len(manifest["ground_truth"]),
        "macro_f1": {role: metrics[role]["events"]["macro_f1"] for role in metrics},
        "output_dir": str(target),
    }


def _require_iou_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("event_iou_threshold must be a number")
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError(
            f"event_iou_threshold must be finite and in (0, 1], got {value!r}"
        )
    return float(value)


def _require_weights(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} weights not found: {path}")
    digest = sha256_file(path)
    if not isinstance(expected, str) or digest != expected.strip().lower():
        raise ValueError(
            f"{label} weights SHA-256 mismatch: expected {expected!r}, got {digest}"
        )
    return digest


def _run_metadata(
    *,
    role: str,
    weights: Path,
    digest: str,
    settings: object,
    info: BackendInfo,
    manifest: Mapping,
    threshold: float,
) -> dict:
    return {
        "comparison": "coco_pretrained_baseline_vs_e0",
        "model_role": role,
        "checkpoint_path": str(weights),
        "checkpoint_sha256": digest,
        "config_sha256": settings.config_sha256,
        "rules_version": settings.rules_version,
        "confidence_threshold": settings.confidence_threshold,
        "movement_threshold": settings.movement_threshold,
        "nms_iou_threshold": settings.nms_iou_threshold,
        "imgsz": settings.imgsz,
        "split": manifest["split"],
        "dataset_release": manifest["dataset_release"],
        "manifest_sha256": manifest["manifest_sha256"],
        "event_iou_threshold": threshold,
        "ultralytics_version": info.version,
        "selection_bias": SELECTION_BIAS,
    }


def _mapping_payload(
    mapping: BaselineClassMapping, weights: Path, digest: str
) -> dict:
    return {
        "schema_version": "1.0",
        "weights_path": str(weights),
        "weights_sha256": digest,
        "names": {str(index): name for index, name in sorted(mapping.baseline_names.items())},
        "mapping": {str(index): project_id for index, project_id in sorted(mapping.class_map.items())},
        "project_class_names": list(mapping.project_names),
        "dropped_classes": [
            {"baseline_index": index, "baseline_name": name}
            for index, name in mapping.dropped
        ],
    }


def _comparison_payload(
    *,
    mapping: BaselineClassMapping,
    baseline_path: Path,
    baseline_digest: str,
    baseline_info: BackendInfo,
    e0_path: Path,
    e0_digest: str,
    e0_info: BackendInfo,
    settings: object,
    manifest: Mapping,
    threshold: float,
    metrics: Mapping[str, dict],
) -> dict:
    return {
        "schema_version": "1.0",
        "comparison": "coco_pretrained_baseline_vs_e0",
        "split": manifest["split"],
        "dataset_release": manifest["dataset_release"],
        "manifest_sha256": manifest["manifest_sha256"],
        "event_iou_threshold": threshold,
        "selection_bias": SELECTION_BIAS,
        "config": {
            "config_sha256": settings.config_sha256,
            "rules_version": settings.rules_version,
            "confidence_threshold": settings.confidence_threshold,
            "movement_threshold": settings.movement_threshold,
            "nms_iou_threshold": settings.nms_iou_threshold,
            "imgsz": settings.imgsz,
        },
        "models": {
            "baseline": {
                "checkpoint_path": str(baseline_path),
                "checkpoint_sha256": baseline_digest,
                "ultralytics_version": baseline_info.version,
                "class_names": {
                    str(index): name for index, name in sorted(baseline_info.names.items())
                },
                "class_mapping": {
                    str(index): project_id
                    for index, project_id in sorted(mapping.class_map.items())
                },
                "dropped_class_count": len(mapping.dropped),
            },
            "e0": {
                "checkpoint_path": str(e0_path),
                "checkpoint_sha256": e0_digest,
                "ultralytics_version": e0_info.version,
                "class_names": {
                    str(index): name for index, name in sorted(e0_info.names.items())
                },
            },
        },
        "metrics": {role: _metric_summary(values) for role, values in metrics.items()},
    }


def _metric_summary(metrics: Mapping[str, dict]) -> dict:
    events = metrics["events"]
    no_change = metrics["no_change"]
    return {
        "per_status": {
            status: {
                key: events["per_status"][status][key]
                for key in ("tp", "fp", "fn", "support", "precision", "recall", "f1")
            }
            for status in STATUSES
        },
        "macro_f1": events["macro_f1"],
        "change_only_macro_f1": events["change_only_macro_f1"],
        "event_iou_threshold": events["iou_threshold"],
        "no_change": {
            "eligible_pair_count": no_change["eligible_pair_count"],
            "false_alarm_pair_count": no_change["false_alarm_pair_count"],
            "false_alarm_rate": no_change["false_alarm_rate"],
            "false_alarm_pair_ids": list(no_change["false_alarm_pair_ids"]),
        },
    }


def _mapping_csv(mapping: BaselineClassMapping) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=MAPPING_CSV_FIELDS, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    for row in mapping.table_rows():
        writer.writerow(row)
    return buffer.getvalue()


def _comparison_csv(metrics: Mapping[str, dict]) -> str:
    rows: list[tuple[str, str, str]] = []
    for status in STATUSES:
        for key in ("precision", "recall", "f1"):
            rows.append((
                f"{status}_{key}",
                f"{metrics['baseline']['events']['per_status'][status][key]:.6f}",
                f"{metrics['e0']['events']['per_status'][status][key]:.6f}",
            ))
    rows.append(("macro_f1",) + tuple(
        f"{metrics[role]['events']['macro_f1']:.6f}" for role in ("baseline", "e0")
    ))
    rows.append(("change_only_macro_f1",) + tuple(
        f"{metrics[role]['events']['change_only_macro_f1']:.6f}" for role in ("baseline", "e0")
    ))
    rows.append(("no_change_eligible_pairs",) + tuple(
        str(metrics[role]["no_change"]["eligible_pair_count"]) for role in ("baseline", "e0")
    ))
    rows.append(("no_change_false_alarms",) + tuple(
        str(metrics[role]["no_change"]["false_alarm_pair_count"]) for role in ("baseline", "e0")
    ))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("metric", "baseline", "e0"))
    writer.writerows(rows)
    return buffer.getvalue()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


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
        description="Compare a COCO-pretrained baseline against E0 on val (not held-out).",
    )
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--e0-weights", type=Path, required=True)
    parser.add_argument("--e0-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-iou-threshold", type=_iou_threshold, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_comparison(
            baseline_weights=args.baseline_weights,
            baseline_sha256=args.baseline_sha256,
            e0_weights=args.e0_weights,
            e0_sha256=args.e0_sha256,
            config_path=args.config,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            event_iou_threshold=args.event_iou_threshold,
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

    output = Path(result["output_dir"])
    print(
        f"split={result['split']} release={result['dataset_release']} "
        f"pairs={result['pair_count']} "
        f"baseline_macro_f1={result['macro_f1']['baseline']:.6f} "
        f"e0_macro_f1={result['macro_f1']['e0']:.6f} output={output}"
    )
    print()
    print((output / "baseline_class_mapping.csv").read_text(encoding="utf-8"), end="")
    print()
    print((output / "comparison.csv").read_text(encoding="utf-8"), end="")
    print()
    print(f"NOTE: {SELECTION_BIAS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
