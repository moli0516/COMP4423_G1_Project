# Independent no-change evaluation protocol v1

`NoChangeEvaluator().evaluate(results, eligible_pair_ids)`.
Results map pair IDs to event lists. Eligible IDs are unique nonempty strings in a
list; results must contain each eligible pair. Empty eligibility is an error.
Additional noneligible pairs are ignored. For each eligible pair, any known status
added/removed/moved makes one false-alarm pair, regardless of number of events.
Rate = false-alarm pairs / all eligible pairs. Unknown/missing statuses are errors,
even after a change has already been found. Results contain sorted alarm IDs.

Eligibility comes from verified independently captured no-change ground truth,
not from predictions, filenames or simply finding one unchanged object.
A mixed-change pair containing unchanged objects is not eligible.
An explicit empty prediction list counts as no change alarm but does NOT prove
correct detection. Event metrics and detector AP must expose missed objects.
This evaluator requires only status fields and deliberately does not validate boxes;
EventEvaluator and upstream validators handle geometry and event structure.
PR5 has no dependency on PR4, PR3, YOLO or UI; it can branch directly from main.
