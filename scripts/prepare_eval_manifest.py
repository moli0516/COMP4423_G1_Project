"""CLI: build a B evaluation manifest for one split of a released dataset.

Usage:
    python scripts/prepare_eval_manifest.py \
        --pairs data/annotations/pairs.jsonl \
        --splits data/manifests/splits.json \
        --split val \
        --dataset-release <release-name> \
        --output artifacts/eval_manifest_val.json

Split membership comes from --splits; only that split's pairs are validated and
emitted, and other splits are never touched. Images are never decoded, and
--dataset-release must match the release recorded in splits.json.

Exit codes:
    0  manifest written
    1  runtime error
    2  invalid input / refused overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The runner's own load_manifest is the verifier, exactly as the val path uses it.
from evaluate_real_pairs import load_manifest

from deskcheck.evaluation.eval_manifest import build_eval_manifest, publish_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a B evaluation manifest for one split.",
    )
    parser.add_argument("--pairs", type=Path, required=True,
                        help="Released pair annotations (pairs.jsonl)")
    parser.add_argument("--splits", type=Path, required=True,
                        help="Released split manifest (splits.json)")
    parser.add_argument("--split", choices=("val", "test"), required=True,
                        help="Split to select from splits.json")
    parser.add_argument("--dataset-release", required=True,
                        help="Dataset release name recorded in the manifest")
    parser.add_argument("--output", type=Path, required=True,
                        help="New manifest JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        payload = build_eval_manifest(
            args.pairs,
            splits_path=args.splits,
            split=args.split,
            dataset_release=args.dataset_release,
            output=args.output,
        )
        publish_manifest(payload, args.output, verify=load_manifest)
    except FileExistsError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1

    print(f"split={payload['split']} pairs={len(payload['pairs'])} "
          f"release={payload['dataset_release']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
