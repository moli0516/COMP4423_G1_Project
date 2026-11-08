"""Run synthetic PR6 scenarios; no YOLO, real images or model claims."""

import argparse
import copy
import json
import math
from pathlib import Path

from deskcheck.evaluation.evaluation_runner import EvaluationRunner


class ScenarioComparisonService:
    """Explicit fake service with call recording and optional failure injection."""

    def __init__(self, predictions: dict, fail_on: str | None = None) -> None:
        self.predictions = predictions
        self.fail_on = fail_on
        self.calls: list[str] = []

    def compare(self, before_path: Path, after_path: Path, pair_id: str) -> dict:
        self.calls.append(pair_id)
        if pair_id == self.fail_on:
            raise OSError("Synthetic inference failure")
        return {"pair_id": pair_id, "records": copy.deepcopy(self.predictions[pair_id])}


def verify_metrics(result: dict, expected: dict) -> None:
    events = result["metrics"]["events"]
    for status, counts in expected["counts"].items():
        for key, value in counts.items():
            if events["per_status"][status][key] != value:
                raise AssertionError(f"Unexpected {status} {key}")
    for key in ("macro_f1", "change_only_macro_f1"):
        if not math.isclose(events[key], expected[key], abs_tol=1e-12):
            raise AssertionError(f"Unexpected {key}")
    if result["metrics"]["no_change"]["false_alarm_rate"] != expected["false_alarm_rate"]:
        raise AssertionError("Unexpected false_alarm_rate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic scenario checks, not model evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="Optional exact case ID, e.g. 07_no_change_alarm")
    args = parser.parse_args()
    data_path = Path(__file__).with_name("fake_evaluation_cases.json")
    cases = json.loads(data_path.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["case_id"] == args.case]
        if not cases:
            parser.error("Unknown case ID")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for case in cases:
        case_dir = args.output / case["case_id"]
        case_dir.mkdir()
        (case_dir / "scenario.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        service = ScenarioComparisonService(
            case["predictions"], "p_removed" if case["scope"] == "runner-failure" else None
        )
        pairs = [{"pair_id": pair_id, "before_path": f"fake/{pair_id}_before.jpg",
                  "after_path": f"fake/{pair_id}_after.jpg"}
                 for pair_id in case["ground_truth"]]
        result = None
        caught = None
        try:
            result = EvaluationRunner(service).run(
                pairs=pairs,
                ground_truth=case["ground_truth"],
                eligible_no_change_pair_ids=case["eligible_no_change_pair_ids"],
                output_dir=case_dir / "evaluation",
                metadata={"source": "synthetic-scenario", "model_used": False,
                          "case_id": case["case_id"]},
            )
        except (RuntimeError, ValueError) as exc:
            caught = exc
        if case["expected_error"]:
            if caught is None or type(caught).__name__ != case["expected_error"]:
                raise AssertionError(f"{case['case_id']}: expected error not observed")
            if (case_dir / "evaluation").exists():
                raise AssertionError("Failed run published completed outputs")
            if case["scope"] == "preflight-failure" and service.calls:
                raise AssertionError("Preflight failure must not call inference")
            if case["scope"] == "runner-failure" and not isinstance(caught.__cause__, OSError):
                raise AssertionError("Original inference cause was not preserved")
            observed = {"error_type": type(caught).__name__, "message": str(caught),
                        "cause": repr(caught.__cause__), "service_calls": service.calls}
            (case_dir / "observed_error.json").write_text(
                json.dumps(observed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            summary = {"case_id": case["case_id"], "title": case["title"],
                       "check": "passed", "expected_error": case["expected_error"]}
        else:
            if caught is not None:
                raise caught
            verify_metrics(result, case["expected"])
            summary = {"case_id": case["case_id"], "title": case["title"], "check": "passed",
                       "macro_f1": result["metrics"]["events"]["macro_f1"],
                       "false_alarm_rate": result["metrics"]["no_change"]["false_alarm_rate"]}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    (args.output / "scenario_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(summaries)} synthetic scenario checks passed; no model was executed.")


if __name__ == "__main__":
    main()
