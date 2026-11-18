"""Owner A: CLI wrapper for the shared trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeskCheck trainer")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to experiment config JSON (e.g. configs/experiments/baseline.json)")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Output run directory (must not exist or be empty)")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Override dataset.yaml path from the config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from deskcheck.training.trainer import TrainingError, train
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        artifact = train(
            args.config,
            run_dir=args.run_dir,
            dataset_yaml_override=args.dataset,
        )
    except TrainingError as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    print(f"checkpoint: {artifact.checkpoint_path}")
    print(f"checkpoint SHA256: {artifact.checkpoint_sha256}")
    print(f"run_dir: {artifact.run_dir}")
    print(f"artifact metadata: {artifact.run_dir / 'artifact.json'}")
    print(f"resolved config: {artifact.run_dir / 'resolved_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
