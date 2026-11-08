# Event evaluation protocol v1 (proposed team policy)

## Data contract
`EventEvaluator(iou_threshold=0.5).evaluate(predicted, ground_truth)`.
Both inputs: `{pair_id: [event, ...]}` with identical nonempty pair-ID inventories.
A pair with no detections must still appear as `[]`. A missing pair is an error.
Each event requires class_id, status, before_box, after_box; extra analyzer fields
are ignored. IDs must be nonnegative canonical integers; one event/class/pair.
Classes absent on both sides create no event. Boxes are original-image xyxy,
finite, positive-area and nonnegative. Image upper bounds are checked upstream.

## Scoring, independent of inference matching
IoU threshold is a declared evaluation choice, not a course-prescribed threshold.
Same-class predicted/true events have unique candidates under basic scope.
Localization uses every shared present side; all shared-side IoUs must be >= threshold.
No shared side cannot localize. For equal statuses: localization passing yields TP;
otherwise predicted status gets FP and true status gets FN. Wrong status similarly
gives FP predicted and FN true whether or not localization passes. This naturally
handles added/removed versus matched-event presence disagreement.
Inference matching remains same-class with NO spatial gate; do not confuse the two.

## Aggregation
Pool TP/FP/FN across all pairs first, then calculate precision/recall/F1.
Support = TP+FN. Undefined values are zero. Macro-F1 averages the fixed four statuses,
including zero-support statuses. Change-only macro-F1 averages added/removed/moved.
Consequently one perfect represented status alone gives macro-F1=0.25, not 1;
all-empty scenes give 0, not 1. Report per-status support and zero-division policy.
No detector AP, binary change F1 or confidence tuning is implemented here.
Rules/thresholds must be approved before final evaluation and disclosed in reports.

## Ground truth and integration
A's annotation loader or C/B integration adapter materializes image references into
before_box/after_box; do not pass raw annotation object IDs expecting automatic joins.
Do not recompute ground truth from predictions. Distance/rules metadata is upstream.
EvaluationRunner will supply complete split inventories and save provenance later.
