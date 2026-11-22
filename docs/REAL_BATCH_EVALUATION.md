# Jerry-C: Real Batch Evaluation (C-owned integration)

The batch CLI constructs the same real comparison service as the UI and single-pair CLI. It passes that service to B's `EvaluationRunner`. The detector receives only canonical RGB frames. Annotations and no-change eligibility go to the evaluator.

## Inputs

Pass three explicit paths from the repository root:

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_real_pairs.py `
  --config configs/app.json `
  --manifest "C:\private\deskcheck\val_pairs_reviewed.json" `
  --output "runs\real-evaluation-unique-name"
```

- `--config` points to a **real, locally existing** checkpoint and SHA-256,
  the frozen class mapping, model identity and threshold. See
  [REAL_MODEL_INTEGRATION.md](REAL_MODEL_INTEGRATION.md).
- `--manifest` is a reviewed B-event manifest, containing a complete pair inventory,
  exact pair-keyed ground-truth events and a list of independently captured
  no-change controls. The schema below is a **C integration transport format**;
  it is not A's native `pairs.jsonl`. Use the validation helper below for A’s native v1 annotations.
- `--output` must be a **new** directory. No existing result is replaced.

Minimal shape (illustrative only; **do not use this as real gold or approved τ**):

```json
{
  "schema_version": "1.0",
  "dataset_release": "REPLACE_WITH_FROZEN_RELEASE_ID",
  "split": "val",
  "pairs": [
    {
      "pair_id": "p0001",
      "before_path": "data/p0001_before.png",
      "after_path": "data/p0001_after.png"
    }
  ],
  "ground_truth": {
    "p0001": [
      {
        "class_id": 0,
        "status": "moved",
        "before_box": [100, 100, 160, 240],
        "after_box": [180, 100, 240, 240]
      }
    ]
  },
  "eligible_no_change_pair_ids": []
}
```

Relative image paths are resolved against the manifest file's directory,
not the process's current working directory. The manifest must declare
`split` explicitly as `val` or `test`. Use the team's frozen split for a formal run. Missing/duplicate
pairs, mismatched truth inventory, missing files, missing checkpoint and
nonfinite JSON values all cause failure without a substitute result.

## Outputs and provenance

On success B creates **exactly** these files:

- `predictions.json`: one record list per pair, from `ComparisonService`.
- `metrics.json`: B's event-level and no-change metrics **only**;
  Detector-level AP requires separate evaluation before B’s per-class filter.
- `run_metadata.json`: pair inventory, explicit eligible controls, evaluation
  protocol, and recorded `split`, `dataset_release`, config SHA-256,
  checkpoint SHA-256, model origin, rules version, thresholds and manifest SHA-256.

## Prepare A's formal validation manifest

From the checkout, supply paths to the released files and a new output directory.

```bash
python scripts/prepare_real_validation.py --annotations release/pairs.jsonl --splits release/splits.json --output runs/val_manifest.json
python scripts/evaluate_real_pairs.py --config configs/app.json --manifest runs/val_manifest.json --output runs/e0_val
```
The helper uses A's catalog and event conversion, selects the validation IDs, preserves missing sides and carries the independently captured control flags. B requires at least one eligible control with annotated unchanged objects to calculate the no-change rate.

## Formal v1 results

On 22 November 2026 the two commands above ran against the frozen four-pair validation split and A's E0 checkpoint. The eight original images were 4096 × 3072 pixels. Their 44 annotated boxes yielded 23 ground-truth events.

| Validation pair | Annotated changes | E0 result |
| --- | --- | --- |
| G01P1 | None | Reported scissors removed. The after-frame detector output had no scissors. |
| G01P2 | Scissors removed | Correct on all six events. |
| G03P1 | Bottle moved, cup added | Correct on all six events. |
| G03P2 | None, independently captured control | Correct on all five events. |

At event IoU 0.50, the evaluator matched 22 of 23 ground-truth events. Added and moved each had one true positive. Removed had one true positive and one false positive. Unchanged had 19 true positives and one false negative. Four-status macro-F1 was **0.9103** and change-only macro-F1 was **0.8889**. The release designates only G03P2 as an independent control, giving **0 alarms in 1 eligible pair**.

| Input | Measured SHA-256 |
| --- | --- |
| Formal v1 ZIP (356,090,127 bytes) | `0a49f3bc1308f1d5ac292cfb6274c17ef89abcc616dc21c1dd1eea0eeafa63f9` |
| Native annotations `pairs.jsonl` | `066adf1eb1921c14bc7729fe577ef5da489102029d38a512cde33eae9c289588` |
| Native split `splits.json` | `6ff54612563c869dae24482c1cc219432c36c011bd3aef8237e7fe8fda982ebb` |

The model, app configuration, and entry-point parity checks are recorded in [the E0 integration handoff](REAL_MODEL_INTEGRATION.md).
