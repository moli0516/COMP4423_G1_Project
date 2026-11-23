"""Owner A: validation-only threshold tuning for the frozen E0 checkpoint.

Runs YOLO val() across a small grid of (confidence, NMS IoU) combinations
on the VALIDATION split only. Never touches test.

Selects the (conf, iou) pair with the highest F1 on val, tie-breaking on
higher mAP50-95. Records the full grid to a CSV for the report.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CONF_GRID = (0.15, 0.20, 0.25, 0.30, 0.40)
IOU_GRID = (0.35, 0.45, 0.55)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validation-only threshold tuning")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to best.pt")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Path to dataset.yaml")
    p.add_argument("--output-csv", type=Path, required=True,
                   help="Where to write the grid results CSV")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. pip install -e '.[vision]'", file=sys.stderr)
        return 1

    model = YOLO(str(args.checkpoint))

    rows = []
    best = None

    print(f"Tuning over {len(CONF_GRID)} × {len(IOU_GRID)} = {len(CONF_GRID)*len(IOU_GRID)} combinations")
    print(f"Val split only. Never test. imgsz={args.imgsz} device={args.device}")
    print()
    print(f"{'conf':>6} {'iou':>6} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 70)

    for conf in CONF_GRID:
        for iou in IOU_GRID:
            results = model.val(
                data=str(args.dataset),
                imgsz=args.imgsz,
                conf=conf,
                iou=iou,
                split="val",
                device=args.device,
                verbose=False,
            )
            # ultralytics returns a DetMetrics object
            box = results.box
            p = float(box.mp) if box.mp else 0.0
            r = float(box.mr) if box.mr else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            map50 = float(box.map50) if box.map50 else 0.0
            map50_95 = float(box.map) if box.map else 0.0

            rows.append({
                "confidence": conf,
                "nms_iou": iou,
                "precision": p,
                "recall": r,
                "f1": f1,
                "map50": map50,
                "map50_95": map50_95,
            })
            print(f"{conf:>6.2f} {iou:>6.2f} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {map50:>8.4f} {map50_95:>10.4f}")

            # Track best by F1, tie-break by mAP50-95
            score = (f1, map50_95)
            if best is None or score > best["_score"]:
                best = {
                    "_score": score,
                    "confidence": conf,
                    "nms_iou": iou,
                    "f1": f1,
                    "map50_95": map50_95,
                }

    # Write CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Grid results: {args.output_csv}")
    print()
    print("BEST by F1 (tie-break mAP50-95):")
    print(f"  confidence = {best['confidence']}")
    print(f"  nms_iou    = {best['nms_iou']}")
    print(f"  F1         = {best['f1']:.4f}")
    print(f"  mAP50-95   = {best['map50_95']:.4f}")
    print()
    print("Next: update configs/experiments/baseline.json with:")
    print(f"  \"inference\": {{\"confidence_threshold\": {best['confidence']}, \"nms_iou_threshold\": {best['nms_iou']}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
