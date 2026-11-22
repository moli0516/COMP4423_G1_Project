"""Print a compact summary of a training run's artifact.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print training artifact summary")
    p.add_argument("path", type=Path, help="Path to artifact.json")
    p.add_argument("--json", action="store_true", help="Print full JSON instead of summary")
    args = p.parse_args(argv)

    if not args.path.exists():
        print(f"not found: {args.path}", file=sys.stderr)
        return 1

    d = json.loads(args.path.read_text(encoding="utf-8"))

    if args.json:
        print(json.dumps(d, indent=2))
        return 0

    print(f"=== {d.get('experiment_id', '?')} ===")
    print(f"  checkpoint:  {d['checkpoint_path']}")
    print(f"  sha256:      {d['checkpoint_sha256']}")
    print(f"  arch:        {d['architecture']} ({d['pretrained_weights']})")
    print(f"  classes:     {d['class_names']}")
    print(f"  image_size:  {d['image_size']}")
    print()
    print(f"  epochs:      {d['epochs_completed']} / {d['epochs_requested']} (selected: {d['selected_epoch']})")
    print(f"  selection:   {d['checkpoint_selection']}")
    print(f"  optimizer:   {d['optimizer']} lr={d['learning_rate']} wd={d['weight_decay']}")
    print(f"  batch:       {d['batch_size']} seed={d['seed']} device={d['device']}")
    print()
    print(f"  dataset:     {d['dataset_yaml']}")
    print(f"  release:     {d['dataset_release']} (seed {d['split_seed']})")
    print(f"  yaml sha256: {d['training_manifest_sha256']}")
    print()
    print(f"  inference:   conf={d['confidence_threshold']} nms_iou={d['nms_iou_threshold']}")
    print(f"  environment: ultralytics {d['ultralytics_version']} / torch {d['torch_version']} / python {d['python_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
