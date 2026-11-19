# Jerry-C / A / B detector handoff — 2026-11-11

**Status: A has accepted the ownership split in [PR #9](https://github.com/moli0516/COMP4423_G1_Project/pull/9#issuecomment-6799449023); B's threshold approval and the shared-interface merge remain separate and pending.**

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

## Accepted A/C detector ownership (B's threshold decisions still pending)

| Responsibility | Agreed owner | Integration decision |
|---|---|---|
| Capture, annotations, frozen split, training, checkpoint and model metadata | A | A supplies actual files and identifies the training run |
| Model loading, model-specific resize/letterbox reversal, confidence, backend NMS, class map, canonical boxes | A | **A explicitly accepted ownership on PR #9 and will adopt/review/refine C's provisional `UltralyticsDetector` on A's branch. One final production adapter, not two competing defaults.** |
| Canonical RGB and EXIF decode once, same sized photo pair, source bytes, UI/CLI export | C | `ImagePairLoader`; no model-specific resizing in this layer |
| C-to-A immutable frame boundary while PR #8 is not merged | C | A expressly agreed to keep `FrameDetectorAdapter` temporarily. The current C-local frame has only `array` plus derived `width`/`height`; A's proposed shared frame has explicit `array`, `width`, `height`, `source_path=None`. Do not replace the seam before PR #8 merges and both sides switch together. |
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

**Runtime version evidence:** No Ultralytics version was installed or actually exercised in the Jerry-C verification. The optional requirement `ultralytics>=8,<9` is a declaration, **not** a tested pin. Real-mode tests injected an API stub; A should select/pin a version from real checkpoint testing. This was answered directly in [PR #9](https://github.com/moli0516/COMP4423_G1_Project/pull/9#issuecomment-6805949361).

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
5. A has accepted detector ownership; A and C must still verify the final single `ImageFrame` type and implemented A-owned adapter after PR #8 merges. B must approve threshold/evaluation settings before PR #9 can be considered for merging into `main`.

**Not claimed:** actual YOLO model loading, real photo accuracy, merged PR #8,
approved B thresholds, real-model parity or final group acceptance.

## Temporary frame boundary checks — 2026-11-12

`tests/c_release/test_frame_boundary_compatibility.py` adds 16 **STUB_ONLY**
cases. Losslessly encoded non-square RGB PNGs with EXIF 2/4/5/7 pass through
the actual C loader, temporary adapter, unchanged B pipeline and serializer.
Independent NumPy pixel oracles verify oriented dimensions and original-pixel
edge boxes. The decoded-frame path runs with file reopening forbidden.
Other cases reject altered factory pixels/dimensions/source identity, detector
metadata mutation and noncanonical detection return types; actual B Detection
object identity is preserved. A test-only `SimpleNamespace` models constructor
fields; it is not A's shared class and does not prove PR #8 integration.

Local verification: 16 new cases plus the three existing adapter regression
cases passed (19 total). The first attempt had four test-harness failures:
the no-I/O guard caught `numpy.testing`'s lazy package-metadata import. The
guarded assertion now uses the already-loaded `numpy.array_equal`; the guard
remains in place. No production code or existing test was changed.

Evidence is limited to software boundaries with an injected stub. There is
no A checkpoint, YOLO runtime, manual browser acceptance or detector AP result
from these checks. PR #8, the final A-owned adapter, reviewed data and B-approved
thresholds remain separate gates. Transparent inputs continue to be rejected
by C pending an agreed alpha policy.
