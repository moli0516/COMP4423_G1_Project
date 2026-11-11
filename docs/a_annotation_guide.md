# Part A — Annotation guide

Status: **PROPOSED** — to be reviewed alongside `docs/a_contracts_freeze.md`.
Owner: A
Audience: A, B, C when annotating or re-checking pairs.

This guide defines how to annotate bounding boxes, classes, correspondences,
and change statuses for DeskCheck. It is the reference both for manual labels
and for how we judge predicted labels during error analysis.

---

## 0. Golden rules

1. Same camera position, orientation, framing, and image dimensions for both
   images of a pair. If the camera moved, discard the pair — do not annotate.
2. Use the actual image content, not the filenames. Filenames never encode
   status.
3. 3. One instance per class per image (basic scope). If a second instance of the
   same class appears, the pair is out of scope: **re-capture** or exclude the
   scene rather than silently leaving a visible target object unannotated.
4. Exclude objects that are fully occluded, cut off by the image boundary, or
   could plausibly be hidden rather than absent.
5. If a class is absent from both images, it produces **no row** — not an
   "unchanged" record.

---

## 1. Drawing boxes

- Draw a tight axis-aligned box around the visible extent of the object.
  Include the whole object; do not clip wings, handles, or wheels.
- Do not pad the box with background "just to be safe".
- If the object extends outside the frame, discard the object (rule 0.4).
- Box coordinates are floats in original oriented image pixels.
- `x1 < x2`, `y1 < y2`, both inside `[0, W]` / `[0, H]`.
- The validator rejects non-positive-area or out-of-bounds boxes.

---

## 2. Class boundaries (from the freeze doc)

| class_id | name | include | exclude |
|---|---|---|---|
| 0 | bottle | tall rigid container with a neck: water bottle, sauce bottle, spray bottle | cups, thermoses without a neck (a mug is a cup) |
| 1 | cup | short open-top drinking vessel: mug, teacup, paper cup | bottles, glasses with stems taller than wide |
| 2 | keyboard | full or compact typing keyboard with a grid of keys | keypads, remote controls, piano keyboards |
| 3 | mouse | pointing device with buttons under the palm | trackpads, trackballs |
| 4 | book | bound volume with a cover and spine: paperback, hardcover, notebook with bound spine | loose paper, single sheets, spiral notebooks (optional), folders |
| 5 | scissors | hand cutting tool with two finger loops and two blades | knives, utility blades |

When a single physical item could fit two classes, choose the more specific
one and record the ambiguity in the capture notes.

---

## 3. Capture groups and pair IDs

- `capture_group_id`: one value per camera setup session. All before/after
  pairs shot without moving the camera share the same group.
- `pair_id`: one value per before/after pair. Two pairs can share a group.
- Never reuse a `pair_id` across groups.
- If two pairs share an image (e.g. the "after" of pair 1 is the "before" of
  pair 2), the split logic keeps the two pairs in the same split. Annotate
  them normally; the pipeline handles the constraint.

Suggested ID formats:
- `g001`, `g002`, ... for groups
- `p0001`, `p0002`, ... for pairs, continuing across groups
- `img0001`, `img0002`, ... for image IDs, one per file

---

## 4. Correspondences and events

For each pair, for each class:

- Present before and present after: add a `correspondences` row linking the
  before `object_id` to the after `object_id`. Then decide the status:
  - Compute `d = hypot(cx_after - cx_before, cy_after - cy_before) / hypot(W, H)`.
  - `d > tau` → `moved`
  - `d <= tau` → `unchanged`
- Present only before: no correspondence; event `removed`.
- Present only after: no correspondence; event `added`.
- Absent both: no correspondence, no event, no record.

The `tau` value is frozen in the freeze doc after pilot evidence. Until then,
annotators should mark only "definitely moved" / "definitely unchanged" and
leave borderline pairs out of the pilot set used to pick `tau`.

`events` reference manual `object_id`s. `bbox_xyxy` values go in
`manual_before_objects` and `manual_after_objects`, not in `events`.

---

## 5. Independent no-change pairs

A pair is `independently_captured_no_change: true` only if:

- Both photos were taken in one short session with **nothing touched** on the
  desk.
- There is no physical possibility of a target object having moved between
  shots.
- Lighting or shadow variation across the two shots is allowed but must not
  change object positions.

These pairs are the denominator for the NoChangeEvaluator false-alarm rate.
Do not mark "I moved nothing but I am not sure" as no-change. When in doubt,
mark `false` and annotate the actual events.

---

## 6. Movement under lighting change

- Lighting change on its own does not produce `moved`. Only center
  displacement above `tau` does.
- If an object's appearance changes (e.g. a book is now open instead of closed)
  but its center stays within `tau`, the status is `unchanged`.
- If an object's center moves enough to exceed `tau`, status is `moved`
  regardless of appearance.

This is a limited movement rule. Appearance and rotation are not modelled.
Note in the capture notes any pairs where appearance change is the only change.

---

## 7. What to record outside the JSONL

Keep a `data/capture_notes.md` with:
- one line per capture group: what was on the desk, lighting, camera setup
- one line per pair: what was actually moved/added/removed in the real world
- excluded objects and why
- ambiguous class assignments

The `events` in the JSONL should agree with these notes. If they don't, the
notes are wrong or the annotation is wrong; resolve the discrepancy before
freezing the split.

---

## 8. Review checklist (per pair)

Before the pair enters `data/annotations/pairs.jsonl`:

- [ ] Camera did not move between before/after
- [ ] Both images decode to the same `(W, H)`
- [ ] Every visible target object has a box on the side where it appears
- [ ] Every box has a class_id from the six approved classes
- [ ] No object is boxed twice for the same class on one side
- [ ] `correspondences` only link matched same-class objects
- [ ] `events` cover every class present on at least one side
- [ ] Absent-both classes produce no row
- [ ] Status matches `d > tau` for moved vs unchanged
- [ ] `independently_captured_no_change` is `true` only when genuinely untouched
- [ ] Capture notes updated
