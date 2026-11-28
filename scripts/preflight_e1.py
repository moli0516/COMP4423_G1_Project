"""Dry-run E1 plan and preflight; never imports or invokes the ML backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.training.e1_experiment import (
    APPROVAL_TOKEN,
    build_receipt,
    execute_e1,
    resolve_plan,
    write_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--treatment", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--initialization", type=Path, default=None)
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None,
                        help="Authoritative pair annotations JSONL for the dataset export")
    parser.add_argument("--splits", type=Path, default=None,
                        help="Authoritative train/val/test split manifest JSON")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="New, nonexistent directory frozen as the execution output")
    parser.add_argument("--output", type=Path, default=None,
                        help="New non-colliding directory for the receipt")
    parser.add_argument("--execute", action="store_true",
                        help="Execute only from a saved, complete preflight specification")
    parser.add_argument("--specification", type=Path, default=None,
                        help="Saved execution_specification.json from a preflight receipt")
    parser.add_argument("--approval-token", default=None,
                        help=f"Required for execution intent; exact value {APPROVAL_TOKEN}")
    parser.add_argument("--owner-approved", action="store_true",
                        help="Record owner/scientific approval; a local assertion, not a signature")
    parser.add_argument("--six-key-policy-approved", action="store_true")
    parser.add_argument("--control-comparability-approved", action="store_true")
    parser.add_argument("--effective-settings-reviewed", action="store_true")
    parser.add_argument("--backend-defaults-verified", action="store_true")
    parser.add_argument("--downstream-rules-frozen", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.execute:
            if args.specification is None:
                raise RuntimeError("--specification is required with --execute")
            planning_inputs = (
                args.control,
                args.treatment,
                args.baseline,
                args.initialization,
                args.dataset_yaml,
                args.annotations,
                args.splits,
                args.run_dir,
                args.output,
            )
            approval_flags = (
                args.owner_approved,
                args.six_key_policy_approved,
                args.control_comparability_approved,
                args.effective_settings_reviewed,
                args.backend_defaults_verified,
                args.downstream_rules_frozen,
            )
            if any(value is not None for value in planning_inputs) or any(approval_flags):
                raise RuntimeError(
                    "--execute accepts only --specification and --approval-token; "
                    "preflight inputs and approval-record flags are not allowed"
                )
            result = execute_e1(
                args.specification,
                approval_token=args.approval_token or "",
            )
            print(json.dumps({"training_invoked": True, "checkpoint": str(getattr(result, "checkpoint_path", None))}))
            return 0

        if args.specification is not None:
            raise RuntimeError("--specification is only valid with --execute")
        if args.approval_token is not None:
            raise RuntimeError("--approval-token is only valid with --execute")
        if args.output is None:
            raise RuntimeError("--output is required for preflight")
        plan = resolve_plan(
            args.control or ROOT / "configs/experiments/baseline.json",
            args.treatment or ROOT / "configs/experiments/augmentation.json",
            ROOT,
            args.baseline or ROOT / "configs/experiments/baseline.json",
        )
        receipt = build_receipt(
            ROOT,
            plan,
            args.initialization,
            args.dataset_yaml,
            args.output,
            annotations_path=args.annotations,
            split_manifest_path=args.splits,
            run_dir=args.run_dir,
            approval_records={
                "owner_scientific_approval": args.owner_approved,
                "six_key_policy_approval": args.six_key_policy_approved,
                "control_comparability_approval": args.control_comparability_approved,
                "effective_settings_review": args.effective_settings_reviewed,
                "backend_defaults_verified": args.backend_defaults_verified,
                "downstream_rules_frozen": args.downstream_rules_frozen,
            },
        )
        receipt_path = write_receipt(receipt, args.output)
        print(json.dumps({
            "training_performed": False,
            "technical_ready": receipt["technical_ready"],
            "formal_training_approved": receipt["formal_training_approved"],
            "execution_permission": receipt["execution_permission"],
            "readiness": receipt["readiness"],
            "receipt": str(receipt_path),
        }))
        return 0 if receipt["formal_training_approved"] else 2
    except (OSError, ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"E1 preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
