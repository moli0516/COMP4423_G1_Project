# Jerry-C / A / B detector handoff — 2026-11-11

**Status: proposal for A/B confirmation; not a shared-interface merge.**

## Evidence and exact revisions

- PR #8 (A): https://github.com/moli0516/COMP4423_G1_Project/pull/8
  at `29f65797c19379a678dc6ce1b5fc39d1b010ffd3`.
  A proposes `src/deskcheck/domain/image_frame.py` with canonical
  RGB uint8 `array`, `width`, `height`, and provenance-only
  `source_path`; `ObjectDetector.predict(image: ImageFrame)`
  returns B `Detection` values in the original oriented pixel coordinates.
- Existing B implementation on Jerry-C includes
  `DetectionFilter`, `ObjectMatcher`, `ChangeAnalyzer` and
  `EvaluationRunner`; Part C's `ComparisonService` calls those
  implementations rather than duplicating the algorithms.
- PR #8 does **not** yet deliver A's trained `YoloDetector`, checkpoint,
  or confirmed A adapter. A's proposed numerical movement
  threshold is not an unconditional approval from B.
- Jerry-C's `ultralytics_detector.UltralyticsDetector` is a **candidate**
  adapter: it consumes C's immutable canonical frame and maps YOLO results
  to B `Detection`. It has API-stub tests but **no A checkpoint acceptance**.

## Proposed single-owner resolution (A and B must confirm)

| Responsibility | Proposed owner | Integration decision |
|---|---|---|
| Capture, annotations, frozen split, training, checkpoint and model metadata | A | A supplies actual files and identifies the training run |
| Model loading, model-specific resize/letterbox reversal, confidence, backend NMS, class map, canonical boxes | A | **One authoritative production detector adapter after review.** A can adopt/review C's provisional `UltralyticsDetector` or supply its own. C must not ship both as competing production defaults |
| Canonical RGB and EXIF decode once, same sized photo pair, source bytes, UI/CLI export | C | `ImagePairLoader`; no model-specific resizing in this layer |
| C-to-A immutable frame boundary while PR #8 is not merged | C | `FrameDetectorAdapter` is a temporary explicit conversion seam. Remove unnecessary conversion once a single shared `ImageFrame` contract is merged and tested |
| Confidence filter, one-per-class matching, four statuses, accepted movement threshold | B | Reuse B's actual unchanged implementations. A/B approve threshold and NMS ownership |
| Event and no-change evaluation on frozen truth | B | Inject the same production `ComparisonService` into `EvaluationRunner`; do not evaluate through a separate model path |
| Runtime config/provenance | C application, validated against A/B-approved values | `RealSettings.load` is source-bound; an effective field change cannot retain the old `config_sha256` through direct construction or `dataclasses.replace` |

## Current shared pipeline (existing, no new algorithm)

```text
RealSettings.load(config.json, verify .pt SHA256)
  -> ImagePairLoader (EXIF -> RGB canonical oriented pixels)
  -> chosen detector (A-approved; currently C candidate only)
  -> B DetectionFilter -> B ObjectMatcher -> B ChangeAnalyzer
  -> one ComparisonService result, same metadata/config hash
       |-> Streamlit annotated views and full JSON/CSV/PNG
       |-> CLI full JSON/CSV/PNG
       |-> B EvaluationRunner (injected ComparisonService)
```

**Current code**: UI `app/streamlit_app.py` and CLI
`scripts/predict_real_pair.py` both load `RealSettings`,
instantiate the **same provisional `UltralyticsDetector` class** and call
the **same `ComparisonService`**. `EvaluationRunner` accepts that service
as its injected comparison service. There are no alternate C implementations
of B's matching/status logic.

## Required acceptance after A delivers real weights

1. A confirms the six-class model names/IDs, checkpoint SHA and correct
   preprocess/NMS implementation; B confirms the approved movement threshold
   and event semantics before running frozen tests.
2. Run one captured before/after photo pair through UI and CLI under the
   *same* resolved config. Confirm oriented original boxes, class map and
   paired records, **not just counts**.
3. Provide the same frozen config and service to B's
   `EvaluationRunner`; compare records, source hashes and errors.
4. Freeze the checkpoint/config/model version; never substitute fake
   detector output on failure.
5. A and B review adapter ownership and the final single `ImageFrame`
   boundary before merging PR #9 into `main`.

**Not claimed:** actual YOLO model loading, real photo accuracy, merged PR #8,
approved A/B thresholds, real-model parity or final group acceptance.
