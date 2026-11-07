# PR3: filtering and B pipeline integration

## Responsibility
`DetectionFilter(confidence_threshold).apply(list[Detection]) -> list[Detection]`.
Use the same filter instance/configuration on both images before ObjectMatcher.
This module performs confidence filtering and one-instance-per-class selection.
It does NOT implement spatial NMS, load models/images, or set production thresholds.
A's detector adapter owns backend NMS and canonical class/original-pixel conversion.

## Selection rules
- Threshold is explicit and finite within [0,1]; reject booleans.
- Keep confidence >= threshold (equality is kept).
- For each class retain the highest confidence, regardless of box overlap.
- Ties: ascending (x1,y1,x2,y2,detection_id).
- Completely identical keys keep the first input object; value-level output remains identical.
- Return selected original Detection objects in ascending class-ID order.
- Do not mutate caller input; selection state does not survive between calls.

## Validation
All inputs must be Detection instances with canonical nonnegative integer class IDs,
string IDs, finite valid confidence, and finite nonnegative positive-area box geometry.
Validation happens before filtering, so corrupt low-confidence inputs do not hide errors.
The filter does not know image dimensions: loader/adapter and ChangeAnalyzer enforce
upper bounds against actual image size. The filter does not enforce a specific class
count: adapter/config must enforce the approved target-class map.

## Integration contract
raw before/after -> filter each side -> ObjectMatcher -> ChangeAnalyzer.
Existing matcher/analyzer signatures and domain records are unchanged. No new
ComparisonService or UI is implemented here: C integrates these modules later.
Retain raw predictions outside the filter if duplicate diagnostics are needed;
`apply` returns only selected detections and no mutable hidden diagnostic state.

## Consequences, not guarantees
Low-confidence after detection can become a removed prediction, and vice versa.
If both sides are filtered out the class produces no event, not unchanged.
These are inference outcomes, not corrected ground-truth labels. Event/no-change
metrics must expose their errors. Filtered empty output alone cannot establish
that a pair is genuinely unchanged. Confidence is selected on validation; the
movement rule/tau remains separately frozen before test annotation.

## Testing and scope
Unit tests cover filtering, boundary equality, duplicate/tie policies, geometry,
sorting, no mutation and stateless calls. Integration tests use actual PR1/PR2
classes and synthetic Detection inputs; they do not require YOLO or Streamlit.
No dependencies, production configuration, weights, image data or evaluation
metrics are added by this PR. This completes B's core inference logic, not all B work.
