# Jerry-C: Real Batch Evaluation (C-owned integration)

**Status:** C entrypoint implemented, tested with an injected YOLO API stub; **not** a trained-model or held-out test result. A owns the final production detector; B owns all evaluation semantics.

## What this actually adds

`scripts/evaluate_real_pairs.py` is the executable bridge from a reviewed photo-pair
inventory to **B's existing** `EvaluationRunner`. It loads the same
`RealSettings`, C's currently provisional `UltralyticsDetector`,
`ComparisonService` and B `EvaluationRunner` as the rest of C. No alternative
status classifier or event scorer has been implemented.

This is a **batch evaluation entrypoint**, not a training script or a change to
A's annotations/exporter. It does not download a checkpoint, use filename clues
for inference, or silently fall back to fake detections.

## Inputs

Pass three explicit paths from the `Jerry-C` repository:

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_real_pairs.py `
  --config "C:\private\deskcheck\approved_real.json" `
  --manifest "C:\private\deskcheck\val_pairs_reviewed.json" `
  --output "runs\real-evaluation-unique-name"
```

- `--config` points to a **real, locally existing** checkpoint and SHA-256,
  the frozen class mapping, model identity and threshold. See
  [REAL_MODEL_INTEGRATION.md](REAL_MODEL_INTEGRATION.md). Ultralytics/PyTorch
  are optional dependencies and have not been installed/verified for this work.
- `--manifest` is a reviewed B-event manifest, containing a complete pair inventory,
  exact pair-keyed ground-truth events and a list of independently captured
  no-change controls. The schema below is a **C integration transport format**;
  it is not A's native `pairs.jsonl`. When A's annotation converter and dataset
  are available, adapt them to this format with A/B agreement, without leaking
  gold labels to the detector.
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
`split` explicitly as `val` or `test`. B's accepted rule/threshold and A's
frozen dataset must be decided before any final `test` run; this CLI never
approves itself to use a held-out test or selects a threshold. Missing/duplicate
pairs, mismatched truth inventory, missing files, missing checkpoint and
nonfinite JSON values all cause failure without a substitute result.

## Outputs and provenance

On success B creates **exactly** these files:

- `predictions.json`: one record list per pair, from `ComparisonService`.
- `metrics.json`: B's event-level and no-change metrics **only**;
  **not detector AP**, precision/recall measured on real images or model accuracy
  until genuine frozen data/checkpoint are supplied.
- `run_metadata.json`: pair inventory, explicit eligible controls, evaluation
  protocol, and recorded `split`, `dataset_release`, config SHA-256,
  checkpoint SHA-256, model origin, rules version, thresholds and manifest SHA-256.

B's evaluator is the only scoring implementation. It receives gold records
**after** the independently decoded pair pixels are passed to the detector.
UI, single-pair CLI and batch runner use the same C `ComparisonService` and B
filter/matcher/change analyzer. After A merges the official detector, that
one backend must replace C's provisional adapter consistently in all three
entrypoints.

## Current validation and handoff

`tests/c_real/test_real_batch_evaluation.py` invokes the actual
`EvaluationRunner` through a deliberate in-memory YOLO API stub, verifies
the two-pair change/control output files and provenance, rejects a missing
checkpoint, and refuses missing truth and overwrites. These tests verify
software wiring, **not** actual YOLO runtime, classroom marking or final
real-photo equivalence.

**Next real acceptance:** after A supplies trained checkpoint, real frozen
pairs and the shared ImageFrame: run the same pair through UI, single-pair CLI
and batch evaluation, compare complete records and hashes, and ask A/B to
review original-image coordinates and frozen threshold. Until then, no
real-photo model metric or final acceptance is claimed.
