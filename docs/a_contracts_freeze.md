# Part A — Contracts freeze (proposal)

Status: PROPOSED — awaiting B's confirmation before merge.
Owner: A
Milestone: catch-up for the 4–6 Nov "approve contracts" step.
Related: docs/member_a_start_here.md, docs/part_b_filtering.md, docs/part_b_rules.md, 02_system_architecture.pdf.

This document freezes the interfaces and semantics that the rest of the
pipeline depends on. It does NOT set numerical training hyperparameters,
the confidence threshold, the NMS IoU, or the movement threshold value —
those are decided by the shared baseline and by pilot evidence (see section 3).

---

## 1. Class list (proposed — 6 classes)

| class_id | name | boundary notes |
|---|---|---|
| 0 | bottle | tall rigid container with a neck; not a cup |
| 1 | cup | short open-top drinking vessel; no neck |
| 2 | keyboard | full-size or compact typing keyboard |
| 3 | mouse | pointing device with buttons |
| 4 | book | bound volume with a cover; not loose paper |
| 5 | scissors | hand cutting tool with two blades |

Scope rules (basic assignment scope):
- At most one instance per class per image.
- A class present in both images of a pair is assumed to be the same physical object.
- Unsupported in this release: pen, phone, mug-as-cup ambiguity, stacked books.
  If a backend returns an unsupported class, the adapter drops it and records the
  drop in run metadata; no silent relabeling.

Decision needed from B: confirm the six classes and the class_id order, or
amend before dataset_validator.py (already coded) hard-codes them.

---

## 2. Status vocabulary (proposed — 4 statuses)

Matches src/deskcheck/domain/change_status.py.

| status | condition | indicates_change |
|---|---|---|
| added | before absent, after present | true |
| removed | before present, after absent | true |
| moved | both present, normalized center distance d > tau | true |
| unchanged | both present, d <= tau | false |

- A class absent from BOTH images produces NO record (not "unchanged").
- Equality d == tau is unchanged.
- The same definition applies to manual annotations, detector output, and
  evaluation.

Decision needed from B: confirm four statuses (change_status.py already has
them). If B prefers collapsing unchanged out of explicit records, say so now —
dataset_validator.py currently requires all four in val/test.

---

## 3. Movement threshold tau — selection procedure (value pending)

## 3. Movement threshold τ — selection procedure and pilot evidence

### 3a. Selection procedure

- τ will be chosen from non-test pilot pairs only.
- Pilot includes small / medium / large movements plus independent no-change
  pairs.
- The selection criterion is "τ that best separates visually-moved from
  visually-unchanged on pilot", documented with the pilot pair list.
- τ will be frozen **before test annotation begins** and will not be revised
  using validation or test results.
- **B owns the final confirmation of τ.** A supplies the pilot evidence.

### 3b. Pilot evidence (11 Nov 2026)

Six pilot pairs were captured on a single desk with one fixed camera
position and stable lighting. Twelve photographs total. The independent
no-change control pair (P6) shows zero measurable drift — camera stability
confirmed.

| Pair | Change | Movement magnitude | Estimated d (normalized) | Verdict |
|---|---|---|---|---|
| P1 | bottle moved ~3 cm | small | ≈ 0.02–0.03 | clearly moved |
| P2 | cup moved ~10 cm diagonal | medium | ≈ 0.06–0.08 | clearly moved |
| P3 | scissors moved left→right of desk | large | ≈ 0.35–0.50 | clearly moved (far) |
| P4 | mouse removed | n/a | n/a | removed |
| P5 | book added | n/a | n/a | added |
| P6 | nothing touched | none | ≈ 0 (control) | unchanged |

**Camera jitter floor** (P6): below measurable threshold (~0).
**Smallest genuine movement** (P1): d ≈ 0.02–0.03.
**Largest control jitter** (P6): ≈ 0.

### 3c. Frozen τ (confirmed by B)

**τ = 0.02152951369355096** (normalized center displacement).

B measured from the 12 pilot boxes (all 24 boxes per class across the 6 pilot pairs) and selected τ as the midpoint between:
- Maximum normalized displacement among untouched (no-change) objects
- Minimum normalized displacement among reportable movements

Scope: 6 pilot pairs from one continuous capture session. Not a real-detector or cross-session validation result.

Movement rule: moved iff d > τ; d ≤ τ is unchanged.

### 3d. Pilot pair mapping (for reproducibility)

| Pair | Before file | After file |
|---|---|---|
| P1 | p1_smallMove1.jpeg | p1_smallMove2.jpeg |
| P2 | p2_midMove1.jpeg | p2_midMove2.jpeg |
| P3 | p3_bigMove1.jpeg | p3_bigMove2.jpeg |
| P4 | p4_sub1.jpeg | p4_sub2.jpeg |
| P5 | p5_add1.jpeg | p5_add2.jpeg |
| P6 | p6_control1.jpeg | p6_control2.jpeg |

Source: `~/Documents/CV Group/pilot_photos/` (not committed; pilot data is
for threshold calibration only, not for training or evaluation).

---

## 4. Manifest schemas (proposed)

### 4a. Pair annotation record — data/annotations/pairs.jsonl, one JSON per line

Example record:

    {
      "schema_version": "1.0",
      "pair_id": "p0001",
      "capture_group_id": "g001",
      "before_image_id": "img0001",
      "after_image_id": "img0002",
      "before_image_path": "data/raw/g001/before.jpg",
      "after_image_path": "data/raw/g001/after.jpg",
      "width": 1920,
      "height": 1080,
      "rules_version": "1.0",
      "independently_captured_no_change": false,
      "manual_before_objects": [
        {"object_id": "o1", "class_id": 0,
         "bbox_xyxy": [100.0, 200.0, 180.0, 400.0]}
      ],
      "manual_after_objects": [
        {"object_id": "o2", "class_id": 0,
         "bbox_xyxy": [140.0, 210.0, 220.0, 410.0]}
      ],
      "correspondences": [
        {"before_object_id": "o1", "after_object_id": "o2", "class_id": 0}
      ],
      "events": [
        {"class_id": 0, "status": "moved",
         "before_object_id": "o1", "after_object_id": "o2"}
      ]
    }

Rules:
- bbox_xyxy in original oriented image pixels, floats, 0 <= x1 < x2 <= W,
  0 <= y1 < y2 <= H.
- correspondences list matched same-class objects. In basic scope each class
  has at most one instance per image, so correspondences are 1:1 per class.
- events use the four statuses in section 2. A class absent from both images
  has no event row.
- independently_captured_no_change is true only for pairs collected
  specifically as no-change controls.

### 4b. Split manifest — data/manifests/splits.json

Example:

    {
      "schema_version": "1.0",
      "seed": 42,
      "dataset_release": "v1",
      "splits": {
        "train": ["p0001", "p0002"],
        "val": ["p0101"],
        "test": ["p0201"]
      },
      "coverage": {
        "classes_per_split": {
          "train": [0, 1, 2, 3, 4, 5],
          "val":   [0, 1, 2, 3, 4, 5],
          "test":  [0, 1, 2, 3, 4, 5]
        },
        "statuses_per_split": {
          "val":  ["added", "removed", "moved", "unchanged"],
          "test": ["added", "removed", "moved", "unchanged"]
        },
        "independent_no_change_pairs": {"val": 3, "test": 4}
      }
    }

The coverage block is optional output for auditors; the validator recomputes
it from the annotations.

### 4c. Capture-group rules

- Every image belongs to exactly one capture_group_id.
- Both images in a pair share the same capture_group_id.
- All pairs that share an image or a capture_group_id must go to the same split.
- Target 60/20/20 by pair count, but group integrity overrides exact ratios.
- Augmentation is applied only to training data after split.

---

## 5. Detection schema (frozen; already in repo)

From src/deskcheck/domain/detection.py:

    @dataclass(frozen=True)
    class Detection:
        detection_id: str
        class_id: int              # nonnegative, canonical from section 1
        confidence: float          # finite, 0 <= c <= 1
        box: BoundingBox           # original-image pixels, positive area

BoundingBox (src/deskcheck/domain/bounding_box.py) validates finite
coordinates and positive area within (width, height).

Contract with B's DetectionFilter: the adapter must
- return canonical class_id in 0..5 (drop unsupported backend classes),
- apply backend NMS BEFORE returning (B's filter does not do spatial NMS),
- undo all resize/letterboxing so box is in original oriented pixels,
- return confidence as a finite float in [0, 1].

DetectionFilter behaviour owned by B (from docs/part_b_filtering.md):
(x1, y1, x2, y2, detection_id), stateless.

---

## 6. ObjectDetector signature (proposed change)

Current repo version:

    def predict(self, image_path: Path) -> list[Detection]: ...

B has asked to switch to:

    def predict(self, image: ImageFrame) -> list[Detection]: ...

`ImageFrame` is now implemented at `src/deskcheck/domain/image_frame.py`
as a frozen dataclass:

```python
@dataclass(frozen=True)
class ImageFrame:
    array: np.ndarray         # RGB uint8, shape (H, W, 3)
    width: int
    height: int
    source_path: Path | None = None   # provenance only, never a feature
```

Ownership policy (agreed with B):
- `array` is canonical, orientation-normalized RGB uint8 data with shape
  (H, W, 3). Loader does NOT resize, crop, or letterbox.
- `width` and `height` are positive and must match `array.shape`.
- `source_path` is provenance only. The detector must work with
  `source_path=None` and must NOT reopen the file.
- Consumers must not mutate the shared array in place; rendering uses
  `.copy()`.

C owns loading/canonicalization. A owns model-specific preprocessing inside
the detector adapter and returns boxes in ImageFrame coordinates.

`yolo_detector.py` is still unimplemented pending the ultralytics pin and
a real checkpoint, but the interface contract is now frozen.

Until B confirms, yolo_detector.py stays unimplemented.

---

## 7. What is NOT frozen by this document

- The numerical value of tau (set from pilot evidence; see section 3).
- The confidence threshold (selected on validation by the shared baseline).
- Backend NMS IoU threshold (selected on validation; A's adapter applies it).
- Input resolution / training hyperparameters (E0 baseline decides; B and C
  vary exactly one factor each for E1 and E2).
- The final class list, statuses, or ImageFrame location — pending B's
  confirmation on sections 1, 2, and 6.

---

## 8. Still missing (data collection)

No real captures exist yet as of 10 Nov 2026:
- No pilot photographs (needed for tau selection; target this week).
- No annotated data/annotations/pairs.jsonl.
- No data/manifests/splits.json.
- No artifacts/best.pt or metadata.

Data pipeline code (src/deskcheck/data/*.py) is written and unit-tested against
synthetic fixtures; it will run unchanged on real data once collected.

---

## 9. Open questions for B (summary)

1. Confirm class list (section 1) and class_id order, or amend.
2. Confirm four statuses (section 2), or collapse unchanged.
3. Approve the tau selection procedure (section 3); any preferred search range?
4. Approve the annotation schema fields (section 4a) — anything extra needed
   for B's EventEvaluator / NoChangeEvaluator?
5. Confirm Detection field names stay as in section 5 (no rename to bbox_xyxy).
6. Confirm the `ImagePairLoader` ownership (§6): C or A?

Items 1, 2, and 6 block downstream code. Items 3–5 are confirmations of things
already coded; changes are cheap now and expensive after pilot annotation.
