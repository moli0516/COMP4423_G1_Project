"""Offline PR6 demonstration: synthetic events, no model or real images."""

import argparse
from pathlib import Path

from deskcheck.evaluation.evaluation_runner import EvaluationRunner


class FakeComparisonService:
    """Test double only; must never be mistaken for a trained image model."""

    def compare(self, before_path: Path, after_path: Path, pair_id: str) -> dict:
        return {"pair_id": pair_id, "records": [{
            "class_id": 0, "status": "unchanged",
            "before_box": [0, 0, 2, 2], "after_box": [0, 0, 2, 2],
        }]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic runner demo, not real evaluation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = {"demo": [{"class_id": 0, "status": "unchanged",
                       "before_box": [0, 0, 2, 2], "after_box": [0, 0, 2, 2]}]}
    result = EvaluationRunner(FakeComparisonService()).run(
        pairs=[{"pair_id": "demo", "before_path": "fake_before.jpg",
                "after_path": "fake_after.jpg"}],
        ground_truth=truth,
        eligible_no_change_pair_ids=["demo"],
        output_dir=args.output,
        metadata={"source": "synthetic-demo", "model_used": False},
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()
