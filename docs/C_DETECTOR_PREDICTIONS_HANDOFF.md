# C → A/B: Retained detector predictions for future AP evaluation

**Status: ADDITIVE DRAFT CONTRACT — A/B review required before merge.**

**Origin:** COMP4423 course brief Task 4 requires mAP@0.5, mAP@[0.5:0.95] and per-class detector results on the held-out test set. The team proposal `05_required_api_list.pdf` proposes retaining before/after predictions, not only change events. A owns detector AP; C only exposes the predictions already used by the app.

## Actual C change (Jerry-C)

`ComparisonService.compare_frames` now uses the **same filtered B predictions**
for both event classification and transport. It returns additive fields
without changing the B modules or the existing event record keys:

```json
{
  "schema_version": "1.0",
  "pair_id": "...",
  "image_size": [160, 120],
  "before_detections": [
    {"detection_id": "example", "class_id": 0,
     "confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 25.0, 25.0]}
  ],
  "after_detections": [],
  "records": [],
  "metadata": {"prediction_scope": "post_B_filter"}
}
```

* `before_detections` / `after_detections` contain **post-B DetectionFilter**
  results in canonical class IDs, confidence and original oriented pixel `xyxy`.
  They are **not raw model outputs**; at most one detection per class remains.
* All fields are predictions. They contain no labels, ground truth or
  dataset-derived hints. No detector AP is calculated by this C code.
* The serializer checks both arrays, unique IDs/classes, finite confidences
  and within-image geometry. Older event-only manually constructed result
  dictionaries still serialize.
* UI and single-pair CLI use `ResultSerializer.to_json`; the full JSON
  includes these fields. The event CSV intentionally remains unchanged.
* B's `EvaluationRunner` currently persists only `records`, so A's
  eventual detector metric tool must consume `ComparisonService`'s
  full result, or agree a shared output artifact/runner extension.
  C does **not** modify B's evaluator on its own.

## Acceptance / remaining work

- Local scope verified using real upstream B filter with duplicates and
  sub-threshold detections: only selected predictions are exported.
- A must confirm that post-B-filter predictions are appropriate for the
  course detector evaluation protocol and retain/integrate them into A's AP
  scorer. B must approve the additive result schema for team merge.
- After A's trained checkpoint and approved `ImageFrame` arrive, verify
  actual original-pixel boxes and class mapping; do not claim the present
  synthetic/stub output demonstrates real detector AP.
- Only after A/B sign-off, freeze schema and update production docs/fixtures
  and group report. No change to A's detector or B's scoring is authorized here.
