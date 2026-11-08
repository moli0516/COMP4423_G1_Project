# Part B rules (PR2)

## Scope and interface
`ChangeAnalyzer(movement_threshold).analyze(matches, image_size)` retains the
existing runtime call signature; annotations are narrowed to matcher output.
Input is already filtered and matched. This module does not load images, apply
confidence thresholds, perform NMS, associate objects or train a model.
`image_size` is a tuple `(width, height)` of positive Python integers for both
original oriented images. The caller checks image alignment/equal dimensions.

## Preconditions
Each match has two sides (Detection or None), at least one present. Both present
must have the same class_id. Each class appears at most once across all matches.
Boxes are validated against original dimensions, including unmatched detections.
Violations raise ValueError and do not return a partial result. Do not skip failed
test samples. Absent in both images means no match/record; `(None,None)` is invalid.

## Presence-first decision table
- Before absent, after present: added (change).
- Before present, after absent: removed (change).
- Both present: compute normalized center distance d.
- d > threshold: moved (change); d <= threshold: unchanged (no change).
- `d = hypot(cx_after-cx_before, cy_after-cy_before) / hypot(width,height)`.
No rounding, tolerance, distance matching gate or IoU association gate is applied.
Only center displacement is modeled: size/rotation/appearance-only changes can
remain unchanged. Basic scope assumes matching classes are the same physical item.

## Threshold ownership
No approved numerical movement threshold is supplied. The caller supplies a finite
number in [0,1]. Unit-test thresholds are synthetic examples, not dataset settings.
Approve the rule and threshold using non-test pilot evidence before test annotation;
use the same definition for manual and predicted boxes. No test-informed revision.

## Event dictionary fields
- class_id: canonical class ID.
- status: JSON string added/removed/moved/unchanged.
- before_detection_id / after_detection_id: side IDs or null if absent.
- before_box / after_box: new xyxy lists or null if absent.
- distance_norm: float for matched objects, null for unmatched.
- movement_threshold: configured numeric threshold (evidence).
- indicates_change: boolean.
Records are sorted by class ID. Input objects/lists are unchanged.

## Integration boundary with C
C's ComparisonService wraps these object-level records with pair_id, image size,
rules_version and artifact/config provenance. Those fields are intentionally not
fabricated here; complete export remains C's responsibility. If pair_id or typed
ChangeRecord input/output is later introduced, agree the interface separately.
No new shared domain type or dependency is introduced in PR2.

## Tests
Tests cover four statuses, exact threshold with 6x8 image (diagonal 10) and center
shift (3,4) (distance 5), below/above boundary, original-coordinate normalization,
far movement, absent sides, invalid dimensions/boxes/classes, JSON output and no
mutation/state leakage. Synthetic inputs isolate rule behavior from detector error.
