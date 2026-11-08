# PR6: injectable evaluation runner

## Scope
Adds orchestration and JSON outputs only. No dataset parser, YOLO adapter, detector
AP, real UI or replacement of scripts/evaluate.py. PR4 and PR5 must already exist.
No third-party dependencies are introduced. One production class in one file.

## API
`EvaluationRunner(comparison_service, iou_threshold=0.5).run(pairs, ground_truth,
eligible_no_change_pair_ids, output_dir, metadata=None)`.
`pairs`: nonempty list of unique {pair_id,before_path,after_path} records; paths
are strings/Path objects, converted to Path for compare. Paths resolve according
to the service/current working directory; the runner does not check image files.
Ground truth has exactly the same pair inventory and materialized event boxes,
not raw object/image references. An upstream adapter must resolve annotations.

## Service contract proposed for C
`compare(before_path=Path(...), after_path=Path(...), pair_id=...)` returns a dict
with identical pair_id and `records` event list. Extra top-level result fields are
not persisted here. All per-event fields are preserved if strictly JSON-safe.
Inference receives only paths and pair ID, NEVER ground truth/no-change eligibility.
Paths/ID must not be used by the model as event features. No inheritance is required;
a fake or real object with a compatible compare method is injected.

## Complete-inventory and error policy
Validate pair IDs, truth, eligibility, caller metadata and output target first.
Use PR4/PR5 public APIs for validation rather than coupling to private helpers.
Every eligible no-change pair must exist, contain annotated objects, and have only
unchanged ground-truth events. Eligibility itself is the caller's assertion of
independent capture; this cannot be verified from event dictionaries.
Process pairs sorted by ID, once each. Validate/snapshot each returned record list
immediately so shared service buffers cannot corrupt earlier results.
Service/output failures raise RuntimeError naming the failed pair and retain
__cause__. No skip-and-continue, fallback predictions or partial summary.
Input/preflight errors retain TypeError/ValueError; filesystem errors propagate.
NoChangeEvaluator's nonzero-denominator policy remains mandatory for this runner.

## Output and publication
Fresh output directory only; never intentionally overwrite an existing file/dir.
- predictions.json: {pair_id: events}.
- metrics.json: {events: PR4 metrics, no_change: PR5 metrics}.
- run_metadata.json: inventory/paths, eligible IDs, timestamps, scoring settings,
  metric scope and caller-supplied provenance.
Strict JSON: no NaN, Infinity, Path objects or automatic string conversion in metadata.
All payloads are serialized before publication. Files are written to a sibling
staging directory and renamed into place after successful writes. Staging is cleaned
on normal exceptions/interruption. Parent directories may remain; abrupt process or
machine failure may leave staging. Do not use concurrent runs targeting the same
output path; this is not a multi-process locking protocol or power-loss guarantee.
Use a unique run directory. No directory of completed results is published on an
inference/evaluation/write failure. Failed runs are reported via exceptions, not a
successful metrics file. Model/data hashes are never invented.

## What this proves / does not prove
Fake-service tests exercise the real event/no-change evaluators and output handling.
They do not prove model accuracy, real image alignment, split isolation or actual
C-service compatibility. Caller must supply real checkpoint/config/rules/dataset
provenance later. Final evaluation needs frozen settings, actual held-out images
and detector AP from A. C must confirm the service return contract before integration.
