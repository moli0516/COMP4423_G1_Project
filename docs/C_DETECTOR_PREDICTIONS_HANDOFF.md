# C → A/B: Retained application diagnostics

**Status: ADDITIVE DRAFT CONTRACT — A/B review required before merge.**

**Origin:** course brief Task 4 requires detector mAP and per-class results. The
team blueprint `05_required_api_list.pdf` proposes retained detections and an
evaluation protocol; it is not a school-mandated API. The present C arrays are
strictly **post-B-filter application diagnostics**, not detector AP/mAP input.
A owns the detector evaluation boundary (backend NMS before B's one-per-class
filter), scoring and independently reviewed box annotations. No detector AP is
calculated or supported by these diagnostic exports.

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
  "records": [
    {"class_id": 0, "status": "removed", "before_detection_id": "example",
     "after_detection_id": null, "before_box": [10.0, 10.0, 25.0, 25.0],
     "after_box": null, "distance_norm": null, "movement_threshold": 0.05,
     "indicates_change": true}
  ],
  "metadata": {"prediction_scope": "post_B_filter"}
}
```

* `before_detections` / `after_detections` contain **post-B DetectionFilter**
  results in canonical class IDs, confidence and original oriented pixel `xyxy`.
  They are **not raw model outputs**; at most one detection per class remains.
* All fields are predictions. They contain no labels, ground truth or
  dataset-derived hints. No detector AP is calculated by this C code.
* The serializer checks both arrays, unique IDs/classes, finite confidences,
  within-image geometry, exact class inventory, ID/box correspondence with
  every event side and null IDs for absent sides. Contradictory full results
  cannot render or export. Older event-only results remain supported.
* UI and single-pair CLI use `ResultSerializer.to_json`; the full JSON
  includes these fields. The event CSV intentionally remains unchanged.
* B's `EvaluationRunner` persists only `records`. These retained arrays are
  **not** in its batch `predictions.json`. A/B must define a separate detector
  prediction artifact/protocol for AP. C does not modify B's evaluator.

## Acceptance / remaining work

- Local scope verified using real upstream B filter with duplicates and
  sub-threshold detections: only selected predictions are exported.
- A/B must review the additive diagnostic schema and define the separate
  detector AP artifact. Never relabel post-B-filter selections as AP input.
  This correction supersedes the older proposed evaluation wording in the
  team blueprint; it does not change a course requirement or B's algorithm.
- After A's trained checkpoint and approved `ImageFrame` arrive, verify
  actual original-pixel boxes and class mapping; do not claim the present
  synthetic/stub output demonstrates real detector AP.
- Only after A/B sign-off, freeze schema and update production docs/fixtures
  and group report. No change to A's detector or B's scoring is authorized here.
