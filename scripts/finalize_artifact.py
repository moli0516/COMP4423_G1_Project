"""Owner A: add post-hoc provenance annotations to a ModelArtifact.

The trainer (trainer.py) writes a base ModelArtifact without the "frozen
vs training-time" split. This script adds those fields as an explicit,
audited post-hoc step so the provenance is documented and reproducible.

Does NOT modify the checkpoint, results.csv, or any trainer output fields.
Only adds the following optional fields to artifact.json:
  - resolved_config_at_training
  - inference_at_training
  - frozen_note
  - resolved_config_note
  - post_hoc_annotation (metadata about this script run)

Idempotent: re-running on an already-finalized artifact preserves the
existing annotation (does not overwrite).

Usage:
    python scripts/finalize_artifact.py \\
        --artifact runs/experiments/E0_baseline/artifact.json \\
        --training-inference-conf 0.25 \\
        --training-inference-iou 0.45 \\
        --frozen-inference-conf 0.20 \\
        --frozen-inference-iou 0.35 \\
        --frozen-note "Frozen after validation-only threshold tuning"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add post-hoc provenance to a ModelArtifact JSON.",
    )
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--training-inference-conf", type=float, required=True,
                   help="confidence_threshold used DURING training (from config at train time)")
    p.add_argument("--training-inference-iou", type=float, required=True,
                   help="nms_iou_threshold used DURING training")
    p.add_argument("--frozen-inference-conf", type=float, required=True,
                   help="confidence_threshold frozen for final evaluation")
    p.add_argument("--frozen-inference-iou", type=float, required=True,
                   help="nms_iou_threshold frozen for final evaluation")
    p.add_argument("--frozen-note", type=str, required=True,
                   help="human-readable justification for the frozen values")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.artifact.exists():
        print(f"artifact not found: {args.artifact}", file=sys.stderr)
        return 2

    try:
        from deskcheck.training.model_artifact import ModelArtifact
    except ImportError as exc:
        print(f"import error: {exc}", file=sys.stderr)
        return 1

    try:
        a = ModelArtifact.load(args.artifact)
    except (OSError, ValueError, TypeError) as exc:
        print(f"failed to load artifact: {exc}", file=sys.stderr)
        return 1

    if a.post_hoc_annotation is not None:
        print(f"artifact already has post_hoc_annotation at {args.artifact} — skipping (idempotent)")
        return 0

    if abs(a.confidence_threshold - args.frozen_inference_conf) > 1e-9:
        print(
            f"warning: artifact top-level confidence_threshold={a.confidence_threshold} "
            f"differs from --frozen-inference-conf={args.frozen_inference_conf}. "
            f"Proceeding but please verify this is intentional.",
            file=sys.stderr,
        )
    if abs(a.nms_iou_threshold - args.frozen_inference_iou) > 1e-9:
        print(
            f"warning: artifact top-level nms_iou_threshold={a.nms_iou_threshold} "
            f"differs from --frozen-inference-iou={args.frozen_inference_iou}",
            file=sys.stderr,
        )

    training_inference = {
        "confidence_threshold": args.training_inference_conf,
        "nms_iou_threshold": args.training_inference_iou,
    }

    original_resolved = dict(a.resolved_config) if a.resolved_config else {}
    original_resolved["inference"] = dict(training_inference)

    timestamp = datetime.now(UTC).isoformat()

    updated = replace(
        a,
        resolved_config_at_training=original_resolved,
        inference_at_training=training_inference,
        frozen_note=args.frozen_note,
        resolved_config_note=(
            "resolved_config_at_training preserves the config as it existed at "
            "model.train() time (before validation-only threshold tuning). "
            "Top-level confidence_threshold and nms_iou_threshold are the FROZEN "
            "inference values. Both are preserved for audit; do not merge."
        ),
        post_hoc_annotation={
            "applied_at": timestamp,
            "applied_by": "scripts/finalize_artifact.py",
            "reason": (
                "Record frozen inference settings separately from training-time "
                "settings so downstream evaluation can distinguish them."
            ),
            "training_inference_conf": args.training_inference_conf,
            "training_inference_iou": args.training_inference_iou,
            "frozen_inference_conf": args.frozen_inference_conf,
            "frozen_inference_iou": args.frozen_inference_iou,
        },
    )

    tmp = args.artifact.with_suffix(".json.tmp")
    updated.save(tmp)
    tmp.replace(args.artifact)

    print(f"updated artifact: {args.artifact}")
    print(f"  training-time inference: conf={args.training_inference_conf} iou={args.training_inference_iou}")
    print(f"  frozen inference: conf={args.frozen_inference_conf} iou={args.frozen_inference_iou}")
    print(f"  post_hoc_annotation.applied_at: {timestamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
