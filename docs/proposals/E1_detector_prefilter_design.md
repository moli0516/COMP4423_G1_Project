# E1 Pre-Filter Detector Predictions — Design Proposal

**Status: DESIGN ONLY — not implemented. No production code, schema or behaviour
was changed for this proposal. A-side and B-side changes described here require
A/B sign-off before any implementation.**

Reviewed HEAD: `b2b1e5b8af04f0ee8d63658cfee9c10bd5b37e2a` (branch `experiment/E1`).

---

## 1. Problem this design solves

`image_detections.json` is captured after backend confidence/NMS, canonical class
mapping, geometry clamping and B's `DetectionFilter`, which keeps at most one
detection per class per image. Scoring it yields an application-pipeline
diagnostic, not detector precision/recall/F1; that is why the current evaluator
was relabelled in this same change set. A true detector metric is blocked until
pre-filter detections are persisted and approved.

---

## 2. Where raw detections actually exist

| Stage | Location | What exists at that point |
|---|---|---|
| Backend predict (conf + NMS) | `src/deskcheck/detection/yolo_detector.py:127-134` | Ultralytics result after confidence filtering and NMS at the configured operating point |
| Class mapping | `src/deskcheck/detection/yolo_detector.py:162-168` | Non-canonical classes are dropped (`continue`); canonical ids assigned |
| Clamp / degenerate drop | `src/deskcheck/detection/yolo_detector.py:175-181` | Boxes clipped to the image; degenerate boxes dropped |
| Deterministic sort | `src/deskcheck/detection/yolo_detector.py:192-193` | Ordered by class, then descending confidence, then detection id |
| **Candidate save point** | `src/deskcheck/application/comparison_service.py:62` | `raw = self._detector.predict(frame)` — full adapter output, still pre-B-filter |
| B confidence + one-per-class | `src/deskcheck/application/comparison_service.py:63` → `src/deskcheck/detection/detection_filter.py:36,38-41` | `filtered = self._filter.apply(raw)`; keeps one detection per class |
| Matcher / change analysis | `src/deskcheck/application/comparison_service.py:72` | Uses the filtered set only |
| Current transport | `src/deskcheck/application/comparison_service.py:103-104,106` | Saves `selected` (post-filter) with `prediction_scope: post_B_filter` |
| Runner persistence | `src/deskcheck/evaluation/evaluation_runner.py:204-213,253-255,278-279` | Writes `image_detections.json` and the `image_detection_evidence` label |

`compare_frames` therefore already holds the exact object a detector metric needs:
the `raw` list from line 62, before line 63. No new inference is required to
capture it; only transport and persistence.

---

## 3. What "pre-filter" can mean

### Option P1 — adapter output, before `DetectionFilter` (minimal)

- Content: canonical class ids, boxes already clamped, NMS already applied,
  confidence already filtered by the backend, at most *N* detections per class.
- Includes: every canonical-class detection the adapter returned.
- Excludes: non-canonical class predictions (dropped at
  `yolo_detector.py:162-168`), degenerate post-clamp boxes, anything the backend
  suppressed by NMS or by `conf > threshold`.
- Answers: "what P/R/F1 does A's delivered adapter produce at 0.20 / 0.35 for the
  six canonical classes, before B's cardinality policy?"
- Cost: additive transport in C, additive persistence in B. No A-side change.

### Option P2 — backend rows before class mapping (audit extension)

- Content: backend class ids/names, confidence, raw box, plus the mapped/clamped
  values for traceability.
- Includes everything P1 includes **and** non-canonical predictions and the
  clipping delta.
- Excludes only what the backend itself suppressed (below-conf, NMS).
- Requires an A-side interface change: `YoloDetector.predict` currently returns
  only canonical `Detection` objects and silently drops unknown classes at
  `yolo_detector.py:162-168` (the code comment acknowledges a dropped-class count
  could be added).

### Open question (must not be decided here)

Should non-canonical class predictions count as detector false positives?

- If **yes**, the metric must be computed over backend rows (P2), because P1 no
  longer contains them.
- If **no**, P1 is sufficient, but the count of dropped non-canonical predictions
  should still be reported as an audit number so the exclusion is visible.

Information that must be preserved to allow either answer later: backend class
id **and** name, confidence, the raw (unclipped) box, the clipped box, the
class-map version, and a per-image count of dropped/non-mapped predictions.

**Recommendation:** implement P1 first (unblocks true P/R/F1 with no A-side
change) and add the P2 audit columns in the same schema version if A agrees,
without deciding whether they enter the primary metric.

---

## 4. Proposed artifact

**File name:** `detector_predictions_raw.json`, written inside the existing run
directory next to `predictions.json`, `ground_truth.json`, `metrics.json`,
`run_metadata.json` and (when present) `image_detections.json`.

**Scope label (exact string):** `detector_raw_before_B_filter`, recorded inside
the artifact as `scope` and in `run_metadata.json` as
`detector_prediction_evidence`.

**Schema (version 1.0, synthetic example):**

```json
{
  "schema_version": "1.0",
  "scope": "detector_raw_before_B_filter",
  "pipeline_stage": "after_backend_conf_nms_class_map_clamp; before_B_DetectionFilter",
  "created_at_utc": "2026-11-26T00:00:00+00:00",
  "run_id": "val-example-aaaaaaaaaaaa-1b2c3d4e",
  "operating_point": {
    "confidence_threshold": 0.2,
    "nms_iou_threshold": 0.35,
    "source_config_path": "/abs/path/app.json",
    "source_config_sha256": "<64 hex>",
    "backend_threshold_semantics": "backend applies conf > threshold; B filter applies >="
  },
  "identities": {
    "checkpoint_sha256": "<64 hex>",
    "dataset_release": "v1",
    "split": "val",
    "manifest_sha256": "<64 hex>",
    "source_annotations_sha256": "<64 hex>",
    "source_split_manifest_sha256": "<64 hex>",
    "class_map_version": "1.0",
    "ultralytics_version": "<recorded at runtime>"
  },
  "images": [
    {
      "image_id": "G01P1__before",
      "pair_id": "G01P1",
      "side": "before",
      "image_sha256": "<64 hex from the run's pair inventory>",
      "detections": [
        {
          "detection_id": "yolo-0003",
          "class_id": 0,
          "confidence": 0.83,
          "bbox_xyxy": [10.0, 12.0, 40.0, 62.0],
          "box_source": "clipped",
          "backend_class_id": 0,
          "backend_class_name": "bottle",
          "raw_bbox_xyxy": [10.0, 12.0, 40.0, 63.5]
        }
      ]
    }
  ]
}
```

Notes:

- Every field above is predictions/identity only; no labels or dataset-derived
  hints.
- `image_sha256` can only be filled when the caller supplies the image hashes that
  `EvaluationRunner` already accepts; otherwise it must be `null` with an explicit
  `"image_identity": "not provided"` marker rather than an omitted field.
- `backend_*` and `raw_bbox_xyxy` are the P2 audit columns; they are `null` when
  the P1 boundary is used.
- A self-hash inside the file is not authoritative. Instead, `run_metadata.json`
  should gain `detector_predictions_sha256` (hash of the artifact file) and
  `detector_prediction_evidence` (the scope label), so the evaluator can bind the
  artifact to the run instead of trusting a hand-edited file.
- Linking to existing artifacts: `run_id`, `checkpoint_sha256`,
  `dataset_release`, `split`, `manifest_sha256` must equal the corresponding
  fields in `run_metadata.json` / its `provenance`; the evaluator rejects
  mismatches. `image_detections.json` remains unchanged and keeps its
  `post_B_filter` evidence label.

**Versioning:** additive migration only. Version 1.0 is required for a detector
metric. A future 1.1 may add fields; the evaluator accepts 1.x with unknown
additive fields rejected by default (`strict`) unless a documented compatibility
window is approved.

### 4.1 Interaction with failure records and rerun links

- **A failed run must not leave a partial pre-filter artifact.** The existing
  runner publishes `predictions.json`, `ground_truth.json`, `metrics.json`,
  `run_metadata.json` and `image_detections.json` only in one atomic publish step;
  `detector_predictions_raw.json` should join that same step. Raw detections
  therefore live in memory during pair evaluation and are dropped on failure, and
  a failed run directory contains only the failure record.
- **A failure record carries identities, not predictions.** It already records
  run id, manifest/checkpoint/config hashes (when known), source identity, the
  evaluation configuration, the command and a timestamp. It must not embed raw
  or partial detections, otherwise it would become a success-shaped artifact that
  an evaluator could score.
- **A rerun produces a new artifact, not a repaired one.** A later run links to
  the earlier failure record(s) via `rerun_of` and writes its own
  `detector_predictions_raw.json` under its own run id. The link records the
  failure identifier and the free-text reason; it does not transfer, merge or
  "heal" the failed run's data.
- **Evaluators must not treat a rerun as replacing a failed run automatically.**
  Whether the rerun counts is an owner decision. The detector path should require
  a completed run (`status: complete` plus the artifact digest in
  `run_metadata.json`), and comparison output should list linked failures rather
  than silently dropping them.
- **Open technical constraint (not decided):** if a future implementation ever
  writes the artifact incrementally, it needs a completion marker or a digest
  recorded only at successful completion; otherwise "file exists" would no longer
  imply "run completed". This proposal assumes the atomic publish above.

---

## 5. Files and interfaces that would change

| File / interface | Change | Owner / sign-off |
|---|---|---|
| `src/deskcheck/application/comparison_service.py` | Add additive `before_detector_detections` / `after_detector_detections` arrays carrying the line-62 `raw` objects, plus a `detector_prediction_scope` metadata key. Existing `before_detections` / `after_detections` keys, ordering and `prediction_scope` are unchanged. | C implements; **A/B sign-off** because the shared result contract is extended |
| `src/deskcheck/application/result_serializer.py` | Validate and pass through the new arrays for the JSON bundle only (event CSV unchanged). | C; **B sign-off** if B consumes the bundle |
| `src/deskcheck/evaluation/evaluation_runner.py` | When the service supplies raw arrays, write `detector_predictions_raw.json`, and record `detector_prediction_evidence` + `detector_predictions_sha256` in `run_metadata.json`. Also persist the comparison service's per-result `prediction_scope`, which is currently dropped, so an evaluator can cross-check per-pair scope instead of only the run-level label. Never derive raw arrays from `image_detections.json`. | **B sign-off** |
| `src/deskcheck/detection/yolo_detector.py` | Only for P2: expose backend rows (class name, raw box, dropped-class counts) without changing the default `predict` contract. | **A sign-off** |
| New `src/deskcheck/evaluation/detector_metric_evaluator.py` | Detector-only path reading only the raw artifact; reuses the shared matcher. | C proposes; **A/B sign-off** on scope wording |
| New `scripts/evaluate_detector_metric.py` | CLI for the detector-only path (no model loading). | C |
| `docs/C_DETECTOR_PREDICTIONS_HANDOFF.md` | Update the draft contract to reference the new artifact. | C edits after A/B review |
| Tests: `tests/c_fake_first/`, `tests/c_real/test_real_batch_evaluation.py`, `tests/integration/test_evaluation_runner.py` | Add raw-array assertions; keep post-filter assertions. | C |

Refactor to reuse matching code: extract the pure helper
`match_detections_to_ground_truth(detections, ground_truth, iou_threshold) ->
outcomes` from `application_pipeline_evaluator.py` into a shared module (for
example `src/deskcheck/evaluation/detection_matching.py`) with **no behavioural
change**, then let both evaluators call it. This keeps a single matching
implementation, satisfies "reuse existing interfaces", and is covered by moving
the existing matching tests unchanged.

### 5.1 Alternatives for the artifact location (options only — not decided)

| Option | Description | Trade-offs |
|---|---|---|
| L1 — run directory (current recommendation) | `detector_predictions_raw.json` sits beside the run's other artifacts | Same lifecycle, one directory to archive; the run directory grows; deleting a run deletes its detector evidence with it |
| L2 — sibling evidence directory | e.g. `<run-dir>.detector/` keyed by the same run id | Clear separation between event and detector evidence; two paths must be kept together when archiving; more ways to orphan one half |
| L3 — separate evidence store | A dedicated evidence root indexed by run id | Best for long-term retention and access control; needs an index, a lookup rule in the evaluator, and a policy for missing entries |

*Recommendation remains L1 (run directory). It is listed with the alternatives
above only to make the trade-off explicit; nothing is decided here.*

### 5.2 Alternatives for one report versus two (options only — not decided)

| Option | Description | Trade-offs |
|---|---|---|
| R1 — two separate reports (current recommendation) | `application_pipeline_metrics.json` and a future `detector_metrics.json`, each with its own scope label | Impossible to confuse scopes; readers must open two files to compare |
| R2 — one combined report with two labelled sections | A single report containing both blocks | Convenient for review; higher risk that a reader quotes the pipeline numbers as detector numbers despite the labels |
| R3 — a thin index plus two reports | A small manifest listing both reports, their scopes and digests | Keeps outputs separate while making discovery easy; adds one more artifact to maintain |

*Recommendation remains R1 (two separate reports).* These are options for the
owner; no implementation is proposed here.


---

## 6. Detector-only evaluation path

```
scripts/evaluate_detector_metric.py --config <config.json> --run <run-dir> \
    --ground-truth <image_boxes.json> --output <new-dir>
```

- Reads **only** `detector_predictions_raw.json` plus per-image ground truth.
- Rejects, with explicit errors, any artifact whose `scope` is not
  `detector_raw_before_B_filter` — in particular `post_B_filter` artifacts,
  missing scope fields, and the file being absent.
- Verifies `run_id`, `checkpoint_sha256`, `dataset_release`, `split`,
  `manifest_sha256` and the artifact digest recorded in `run_metadata.json`;
  records what it could not verify (checkpoint bytes, upstream re-execution) as
  `not_verified`.
- Uses the shared matcher at IoU 0.5, class-exact, greedy by descending
  confidence, with the same deterministic tie-breaks.
- Writes `detector_metrics.json` with
  `metric_scope: "detector P/R/F1 at the recorded operating point; backend NMS applied, before B cardinality filter"`.
- The application-pipeline report (`application_pipeline_metrics.json`) keeps its
  own scope label and file. The comparator may list both, clearly separated, and
  must never average or merge them.
- Cross-check regression: for one synthetic run, the detector path on raw boxes
  and the pipeline path on post-filter boxes must differ exactly as the
  one-per-class truncation predicts, and the test asserts both numbers.

---

## 7. Test plan (synthetic fixtures only)

1. Schema: valid artifact accepted; unknown scope rejected; missing scope
   rejected; wrong version rejected; duplicate JSON keys rejected; non-finite
   numbers rejected.
2. Binding: mismatched `run_id`, `checkpoint_sha256`, `manifest_sha256`, or
   artifact digest rejected; matching values accepted; `not_verified` list
   recorded.
3. Identity: `image_sha256` mismatch (when provided) rejected; `null` with the
   `image_identity` marker accepted but recorded.
4. Matching: the existing matrix runs unchanged against the detector path
   (perfect, class mismatch, IoU boundary above/at/below, duplicates, ties,
   empty sides, invalid boxes, unknown classes).
5. Two-same-class raw boxes at 0.20/0.35 produce raw recall 1.0 and post-filter
   recall 0.5, proving the two artifacts are different inputs.
6. Confidence: a raw detection exactly at the threshold accepted; strictly below
   rejected as internally inconsistent with the recorded operating point.
7. Storage/round-trip: artifact written by the runner parses back identically;
   `image_detections.json` is unchanged in the same run.
8. Regression: all existing C/B tests still pass; the pipeline report keeps its
   own scope label and is not accepted by the detector path.
9. No synthetic fixture may be named or placed so that it resembles a formal E1
   result.

---

## 8. Risks

- **Schema compatibility:** adding keys to `comparison_service` results and
  `run_metadata.json` can break exact-key-set assertions in C/B tests and any
  strict consumers. Mitigation: additive keys only, versioned evidence labels,
  and a documented strict/loose policy.
- **Storage size:** roughly two arrays per pair instead of one. At the scale of
  24 pairs × 2 images and a handful of detections per image this is well under a
  megabyte of JSON; even a 10× detection count stays small relative to the
  images. No binary format is justified.
- **Effect on existing tests:** several tests assert exact result key sets
  (`tests/c_fake_first/test_c_application.py`, `tests/c_real/test_real_batch_evaluation.py`,
  `tests/c_release/test_retained_detections_contract.py`); they would need
  updating in the same change as the new fields.
- **Effect on E1 evidence format:** `run_metadata.json` gains fields, so
  previously saved runs remain readable but lack the new evidence; the evaluator
  must reject them for a detector metric with a clear message rather than fall
  back to post-filter data.
- **Confusion risk:** two detection artifacts could be mixed up. Mitigation:
  distinct filenames, distinct scope strings, evaluator-side rejection of the
  wrong scope, and an explicit cross-check test.
- **Backend semantics drift:** recorded thresholds describe the request, not the
  backend's internal `>` vs `>=`; the artifact must carry
  `backend_threshold_semantics` as a recorded note, not an assumption.

---

## 9. Sequence (each step independently revertible)

1. **Extract the matcher** into a shared helper with no behaviour change; move
   tests unchanged. *Rollback:* restore the private method and delete the helper.
2. **C: add raw arrays** to `compare_frames` result serialization (additive).
   *Rollback:* remove the additive keys; no consumer yet.
3. **B: persist the artifact** and record evidence + digest in `run_metadata`.
   *Rollback:* stop writing the file and remove the metadata fields; existing
   runs unaffected.
4. **C: add the detector-only evaluator and CLI**, rejecting post-filter scopes.
   *Rollback:* delete module/CLI and tests; no effect on existing artifacts.
5. **Docs and protocol:** update the handoff draft and the E1 protocol only after
   A/B sign-off on the scope wording. *Rollback:* revert the documentation
   commit, which changes no code.
6. **Formal use:** only after sign-off, trusted ground-truth boxes, and
   real-backend verification.

---

## 10. Open owner decisions (none decided here)

| # | Decision | Options | Trade-offs | Recommendation |
|---|---|---|---|---|
| D1 | Detector boundary | P1 adapter output vs P2 backend rows | P1 is cheap and needs no A change but cannot count non-canonical FPs; P2 is more complete but changes A's interface | P1 first; add P2 audit columns if A agrees |
| D2 | Non-canonical class predictions | Count as FP / exclude / report separately | Counting changes the metric definition; excluding hides adapter drops | Report as a separate audit count, exclude from the primary metric until the owner decides |
| D3 | Threshold boundary | Record `>` semantics / normalise to `>=` | The backend and B filter already differ; normalising invents behaviour | Record the semantics string and validate only saved confidences |
| D4 | Artifact location | Run directory / separate evidence directory | Same lifecycle is simpler; a separate directory makes provenance clearer | Run directory, with digest recorded in `run_metadata.json` |
| D5 | Scope approval | A only / B only / A and B | The label defines what the metric means | A and B |
| D6 | Whether both metrics appear on one report page | Separate files/labels / combined summary | A combined summary invites accidental merging | Separate outputs; a comparison may list both with explicit roles |
