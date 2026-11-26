"""Compare two completed evaluation directories without running inference."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.evaluation.run_comparator import compare_runs, write_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.control, args.treatment)
        path = write_comparison(report, args.output)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"Evaluation comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Comparison written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
